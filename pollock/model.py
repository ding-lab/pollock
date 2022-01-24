import logging
import random
import time
from collections import Counter

import anndata
import numpy as np
import scanpy as sc

import torch
import torch.nn.functional as F


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class ZINBLoss(torch.nn.Module):
    """
    Adapted from scDCC https://github.com/ttgump/scDCC/blob/65bcbbd63e2e80785a3f4d9bd8f3cedd8f38f6ca/layers.py
    """
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class MeanAct(torch.nn.Module):
    """
    Pulled from scDCC https://github.com/ttgump/scDCC/blob/65bcbbd63e2e80785a3f4d9bd8f3cedd8f38f6ca/layers.py
    """
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(torch.nn.Module):
    """
    Pulled from scDCC https://github.com/ttgump/scDCC/blob/65bcbbd63e2e80785a3f4d9bd8f3cedd8f38f6ca/layers.py
    """
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class PollockModel(torch.nn.Module):
    def __init__(self, genes, classes,
                 latent_dim=64, enc_out_dim=128, middle_dim=512,
                 zinb_scaler=1., kl_scaler=1e-5, clf_scaler=1.):
        """
        Pollock VAE + classifier
        """
        super(PollockModel, self).__init__()
        self.latent_dim = latent_dim
        self.genes = genes
        self.n_genes = len(genes)
        self.classes = classes
        self.n_classes = len(classes)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_genes, middle_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_dim, enc_out_dim),
            torch.nn.ReLU(),
        )

        self.mu = torch.nn.Linear(enc_out_dim, latent_dim)
        self.var = torch.nn.Linear(enc_out_dim, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, enc_out_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(enc_out_dim, middle_dim),
            torch.nn.ReLU(),
        )
        self.disp_decoder = torch.nn.Sequential(
            torch.nn.Linear(middle_dim, self.n_genes),
            DispAct()
        )
        self.mean_decoder = torch.nn.Sequential(
            torch.nn.Linear(middle_dim, self.n_genes),
            MeanAct()
        )
        self.drop_decoder = torch.nn.Sequential(
            torch.nn.Linear(middle_dim, self.n_genes),
            torch.nn.Sigmoid()
        )

        self.prediction_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, self.n_classes),
            torch.nn.Softmax(dim=1),
        )

        self.zinb_loss = ZINBLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.zinb_scaler = zinb_scaler
        self.kl_scaler = kl_scaler
        self.clf_scaler = clf_scaler

    def kl_divergence(self, z, mu, std):
        # lightning imp.
        # Monte carlo KL divergence
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)

        return kl

    def encode(self, x, use_means=False):
        x_encoded = self.encoder(x)
        mu, log_var = self.mu(x_encoded), self.var(x_encoded)

        # sample z from parameterized distributions
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        # get our latent
        if use_means:
            z = mu
        else:
            z = q.rsample()

        return z, mu, std

    def decode(self, x):
        h = self.decoder(x)
        x_disp = self.disp_decoder(h)
        x_mean = self.mean_decoder(h)
        x_drop = self.drop_decoder(h)

        return x_disp, x_mean, x_drop

    def calculate_loss(self, r, x_raw, scale_factor, y_true):
        reconstruction_loss = self.zinb_loss(
            x_raw, r['x_mean'], r['x_disp'], r['x_drop'], scale_factor=scale_factor)

        kl_loss = torch.mean(self.kl_divergence(r['z'], r['mu'], r['std']))

        clf_loss = torch.mean(self.ce_loss(r['y'], y_true))

        return ((reconstruction_loss * self.zinb_scaler) + (kl_loss * self.kl_scaler) + (clf_loss * self.clf_scaler),
                reconstruction_loss,
                kl_loss,
                clf_loss)

    def forward(self, x, use_means=False):
        z, mu, std = self.encode(x, use_means=use_means)
        x_disp, x_mean, x_drop = self.decode(z)
        y = self.prediction_head(z)

        return {
            'z': z,
            'mu': mu,
            'std': std,
            'x_disp': x_disp,
            'x_mean': x_mean,
            'x_drop': x_drop,
            'y': y
        }


def fit_model(model, opt, scheduler, train_dl, val_dl, epochs=20):
    use_cuda = next(model.parameters()).is_cuda
    history = []
    for epoch in range(epochs):
        train_loss, val_loss = 0., 0.
        val_recon_loss, val_kl_loss, val_clf_loss = 0., 0., 0.
        start = time.time()
        model.train()
        for i, b in enumerate(train_dl):
            x, x_raw, sf, y = b['x'], b['x_raw'], b['size_factor'], b['y']
            if use_cuda:
                x, x_raw, sf, y = x.cuda(), x_raw.cuda(), sf.cuda(), y.cuda()
            opt.zero_grad()
            out = model(x)
            loss, recon_loss, kl_loss, clf_loss = model.calculate_loss(out, x_raw, sf, y)
            loss.backward()
            opt.step()

            train_loss += float(loss.detach().cpu())
            scheduler.step()
        train_loss = train_loss / len(train_dl)

        time_delta = time.time() - start
        model.eval()
        with torch.no_grad():
            for i, b in enumerate(val_dl):
                x, x_raw, sf, y = b['x'], b['x_raw'], b['size_factor'], b['y']
                if use_cuda:
                    x, x_raw, sf, y = x.cuda(), x_raw.cuda(), sf.cuda(), y.cuda()

                out = model(x)
                loss, recon_loss, kl_loss, clf_loss = model.calculate_loss(out, x_raw, sf, y)
                val_loss += float(loss.detach().cpu())
                val_recon_loss += float(recon_loss.detach().cpu())
                val_kl_loss += float(kl_loss.detach().cpu())
                val_clf_loss += float(clf_loss.detach().cpu())

        val_loss, val_recon_loss, val_kl_loss, val_clf_loss = [
            l / len(val_dl) for l in [val_loss, val_recon_loss, val_kl_loss, val_clf_loss]]

        history.append({
            'epoch': epoch,
            'train loss': train_loss,
            'val loss': val_loss,
            'val reconstruction loss': val_recon_loss,
            'val_kl_loss': val_kl_loss,
            'val classification loss': val_clf_loss,
            'time': time_delta
        })
        logging.info(f'epoch: {epoch}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, \
zinb loss: {val_recon_loss:.3f}, kl loss: {val_kl_loss:.3f}, \
clf loss: {val_clf_loss:.3f}, time: {time_delta:.2f}')

    return history

