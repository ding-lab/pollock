import logging
import numpy as np
import pandas as pd

import torch
from captum.attr import DeepLiftShap, IntegratedGradients

from pollock.dataloaders import get_prediction_dataloader


class AttributionWrapper(torch.nn.Module):
    def __init__(self, model):
        super(AttributionWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        result = self.model(x)
        return result['y']


def explain_adata(model, adata, baseline_adata, target, device='cpu',
                  method='integrated'):
    try:
        a_dl = get_prediction_dataloader(adata, model.genes)
        b_dl = get_prediction_dataloader(baseline_adata, model.genes)
        a, ab = a_dl.dataset.adata, b_dl.dataset.adata
    except ZeroDivisionError:
        logging.info(f'explain for target {target} with size of {adata.shape} failed in normalization')
        return None

    X = a.X.toarray() if 'sparse' in str(type(a.X)).lower() else a.X
    baseline_X = ab.X.toarray() if 'sparse' in str(type(ab.X)).lower() else ab.X
    inputs = torch.tensor(X)
    baseline = torch.tensor(baseline_X)

    inputs, baseline = inputs.to(device), baseline.to(device)

    attr_model = AttributionWrapper(model.to(device))

    if not isinstance(target, int) and target != 'all':
        target = model.classes.index(target)

    if method == 'deeplift':
        dls = DeepLiftShap(attr_model)
    elif method == 'integrated':
        baseline = torch.mean(baseline, dim=0).unsqueeze(dim=0)
        dls = IntegratedGradients(attr_model)

    if target != 'all':
        attrs, _ = dls.attribute(
            inputs, baseline, target=target, return_convergence_delta=True)
        return pd.DataFrame(data=attrs.detach().cpu().numpy(),
                            columns=model.genes,
                            index=a.obs.index.to_list())
    else:
        attrs = {}
        for i in range(len(model.classes)):
            attr, _ = dls.attribute(
                inputs, baseline, target=i, return_convergence_delta=True)
            attrs[model.classes[i]] = pd.DataFrame(
                data=attr.detach().cpu().numpy(),
                columns=model.genes,
                index=a.obs.index.to_list())
        return attrs


def explain_predictions(model, adata, adata_background, label_key='cell_type',
                        n_sample=None, device='cpu'):
    attbs = None
    labels = sorted(set(adata.obs[label_key]))
    for label in labels:
        f = adata[adata.obs[label_key]==label]
        if n_sample is not None:
            f = f[np.random.choice(
                      f.obs.index.to_list(), size=min(n_sample, f.shape[0]),
                      replace=False)]

        a = explain_adata(model, f, adata_background, target=label, device=device)
        if a is not None:
            a[label_key] = label

            if attbs is None:
                attbs = a
            else:
                attbs = pd.concat((attbs, a), axis=0)

    return attbs
