import logging
import json
import math
import os
import random
import re
import shutil
import uuid
from collections import Counter

import anndata
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import scanpy as sc
from PIL import Image
from sklearn.metrics import classification_report


from tensorflow.keras.models import Sequential
import tensorflow as tf

import pollock.preprocessing.preprocessing as pollock_pp
import pollock.models.analysis as pollock_analysis


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


AUTOTUNE = tf.data.experimental.AUTOTUNE


MODEL_PATH = 'model.h5' 
GENE_TEMPLATE_PATH = 'gene_template.npy' 
CELL_TYPE_TEMPLATE_PATH = 'cell_type_template.npy' 
CELL_TYPES_PATH = 'cell_types.npy' 
MODEL_SUMMARY_PATH = 'summary.json' 

def set_training_devices(devices, jit=False):
    if jit:
        tf.config.optimizer.set_jit(True)

    tf.distribute.MirroredStrategy(devices=devices)


def get_row_and_col(location, overall_shape, block_shape):
    r = int(location)
    c = int((location - r) * overall_shape[1])
    r = r * block_shape[0]
    
    return r, c


def create_block_image_template(adata, key='ClusterName', block_shape=(4, 4), size=(128, 128), nn_threshold=None):
    n_genes = block_shape[0] * block_shape[1]
    sc.tl.rank_genes_groups(adata, key, n_genes=n_genes)
    ranked_genes_groups = adata.uns['rank_genes_groups']['names']
    cell_types = sorted(set(adata.obs[key]))
    pairs = []

    if nn_threshold is not None:
        logging.info('setting up close groupings')
        logging.info('calculating nearest neighbors')
        sc.tl.pca(adata, svd_solver='arpack')
        #sc.tl.umap(adata)
        sc.pp.neighbors(adata, n_neighbors=15)

        logging.info('calculating connectivities')
        cell_type_to_connectivity = get_connectivities(adata, key)
        close_groupings = get_close_groupings(cell_type_to_connectivity,
                threshold=nn_threshold)

        logging.info('calculating differential genes')
##         grouping_to_differential_genes = get_differential_for_grouping(adata,
##                 close_groupings, n_genes=n_genes)
        gene_to_diffs = get_differential_for_grouping(adata, key, close_groupings,
                n_genes=n_genes)

        pairs = [f'{a}_diff_{b}' for a, b in close_groupings]
        pair_to_genes = {f'{a}_diff_{b}':gs for (a, b), gs in gene_to_diffs.items()}

    combined_labels = cell_types + pairs

    total_pixels = size[0] * size[1]
    current_fraction = int(total_pixels / (4 * n_genes * len(combined_labels)))
    expansions = int(math.log(current_fraction, 4)) + 1

    spots = np.power(4, expansions)
    block = np.full(size, '', dtype=object)
    cell_template = np.full(size, '', dtype=object)
    for i, label in enumerate(combined_labels):
        if label in cell_types:
            genes = ranked_genes_groups[label]
        else:
            genes = pair_to_genes[label]

        bs = np.full((block_shape[0] * int(np.sqrt(spots)), block_shape[1] * int(np.sqrt(spots))), '', dtype=object)
        for j, gene in enumerate(genes):
            b = np.full((spots,), gene).reshape((int(np.sqrt(spots)), int(np.sqrt(spots))))
            location = (j * b.shape[0]) / bs.shape[0]
            r, c = get_row_and_col(location, bs.shape, b.shape)
            bs[r:r + b.shape[0], c:c + b.shape[1]] = np.copy(b)


        location = (i * bs.shape[0]) / size[0]
        r, c = get_row_and_col(location, size, bs.shape)
        block[r:r + bs.shape[0], c:c + bs.shape[1]] = np.copy(bs)
        cell_template[r:r + bs.shape[0], c:c + bs.shape[1]] = np.full(bs.shape, label, dtype=object)

    return block, cell_template

## def get_expression_image(gene_to_expression, template):
##     gene_img = np.full(template.shape, 0, dtype=int)
##     for i in range(template.shape[0]):
##         for j in range(template.shape[1]):
##             gene_img[i, j] = gene_to_expression.get(template[i, j], 0) + 1
##     return np.asarray(gene_img)

## def get_expression(gene, d):
##     return d[gene] + 1
## def get_expression_image(gene_to_expression, template):
##     vf = np.vectorize(get_expression)
##     return vf(template, gene_to_expression)

def get_expression_image(gene_to_expression, template):
    #img = np.full(template.shape, 1., np.float32)
    img = np.full(template.shape, 1, np.uint8)
    for g, e in gene_to_expression.items():
        img[template==g] = e
        #img = np.where(template==g, e, 1)
    return img

def get_connectivities(adata, cell_type_key):
    ## calculate connections
    cell_types = sorted(set(adata.obs[cell_type_key]))
    cell_type_to_idxs = {c:set(np.argwhere(np.asarray(adata.obs[cell_type_key])==c).flatten())
            for c in cell_types}
    cell_type_to_connectivity = {}
    connectivities = adata.uns['neighbors']['connectivities']
    
    # mask = adata.uns['neighbors']['distances'] > 0
    # connectivities = adata.uns['neighbors']['distances']
    # connectivities[mask] = 1
    # connectivities.toarray()
    
    for cell_type_a in cell_types:
        conns = {}
        for cell_type_b in cell_types:
            a_idxs, b_idxs = cell_type_to_idxs[cell_type_a], cell_type_to_idxs[cell_type_b]
                    
            m1 = np.asarray([np.full((adata.shape[0]), True) if r in a_idxs else np.full((adata.shape[0]), False)
                               for r in range(adata.shape[0])])
            m2 = np.asarray([np.full((adata.shape[0]), True) if c in b_idxs else np.full((adata.shape[0]), False)
                               for c in range(adata.shape[0])]).transpose()
            mask = m1 & m2
            local = connectivities[mask]
    
            conns[cell_type_b] = np.sum(local) * (len(a_idxs) / len(b_idxs))
    
        cell_type_to_connectivity[cell_type_a] = conns
        
    cell_type_to_connectivity = {k:{ct:value / sum(v.values()) for ct, value in v.items()}
                                 for k, v in cell_type_to_connectivity.items()}

    return cell_type_to_connectivity

def get_close_groupings(cell_type_to_connectivity, threshold=.05):
    cell_type_to_close = {cell_type:[] for cell_type in cell_type_to_connectivity.keys()}
    tups = []
    for cell_type, d in cell_type_to_connectivity.items():
        for k, val in d.items():
            if val > threshold and k != cell_type: tups.append(tuple(sorted([cell_type, k])))
                
    return sorted(set(tups))

def get_differential_for_grouping(adata, cell_type_key, close_groupings, n_genes=16):
    grouping_to_differential = {}
    for cell_type_a, cell_type_b in close_groupings:
        filtered = adata[(adata.obs[cell_type_key]==cell_type_a) | (adata.obs[cell_type_key]==cell_type_b)]
        sc.tl.rank_genes_groups(filtered, cell_type_key, n_genes=n_genes)
        genes = list(filtered.uns['rank_genes_groups']['names'][cell_type_a][:n_genes // 2])
        genes += list(filtered.uns['rank_genes_groups']['names'][cell_type_b][:n_genes // 2])
        grouping_to_differential[(cell_type_a, cell_type_b)] = genes
    return grouping_to_differential


def initialize_directories(filesafe_cell_types, root=os.path.join(os.getcwd(), 'temp_images'),
        directories=['training', 'validation', 'all'], purpose='training'):
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.mkdir(root)
    for x in directories:
        os.mkdir(os.path.join(root, x))

        if purpose == 'training':
            for cell_type in sorted(set(filesafe_cell_types)):
                os.mkdir(os.path.join(root, x, cell_type))
        else:
            os.mkdir(os.path.join(root, x, 'unlabeled'))

            
def write_images(adata, template, cell_type_to_filesafe, cell_type_key='ClusterName', root=os.path.join(os.getcwd(), 'temp_images'),
        purpose='training', batching_size=1000):
    cell_type_to_fps = {}
    
    logging.info('writing images')
    is_sparse = 'sparse' in str(type(adata.X))
    template = np.asarray(template)
    available_genes = set(template.flatten())
    for b in range(0, adata.shape[0], batching_size):
        logging.info(f'{b} cell images written')
        batching_adata = adata[b:b+batching_size]

        X = batching_adata.X
        if is_sparse:
            X = X.toarray()

        genes, idxs = zip(*[(g, i) for i, g in enumerate(batching_adata.var.index) if g in available_genes])
        X = X[:, np.asarray(idxs)]

        for i in range(batching_adata.shape[0]):
            if purpose == 'training':
                cell_type = batching_adata.obs[cell_type_key][i]
            else:
                cell_type = 'unlabeled'
    
            if cell_type_to_filesafe[cell_type] not in cell_type_to_fps:
                cell_type_to_fps[cell_type_to_filesafe[cell_type]] = []
    
#            logging.info('grab cell id')
            cell_id = batching_adata.obs.index[i]

#            logging.info('creating expression dict')
            expression = (X[i] / np.max(X[i])) * 255.
            gene_to_expression = {g:e
                    for g, e in zip(genes, expression)}
#            logging.info('creating expression image')
            gene_img = get_expression_image(gene_to_expression, template)
#            logging.info('normalizing image rnage')
            #gene_img = (gene_img / np.max(gene_img)) * 255
#            logging.info('set type')
            #gene_img = gene_img.astype(np.uint8)
#            logging.info('swap axes')
            gene_img = np.moveaxis(np.asarray([gene_img, gene_img, gene_img]), 0, -1)
#            logging.info('save image')
            fp = os.path.join(root, 'all', cell_type_to_filesafe[cell_type], f'{cell_id}.jpg')
#            logging.info('append to fp dict')
            cell_type_to_fps[cell_type_to_filesafe[cell_type]].append(fp)
    
            mpl.image.imsave(fp, gene_img)
    logging.info('done writing images')
    return cell_type_to_fps
        
def setup_training(cell_type_to_fps, train_split=.8, n_per_cell_type=200, max_val_per_cell_type=50):
    for cell_type, fps in cell_type_to_fps.items():
        split = int(len(fps) * train_split)

        training = fps[:split]
        validation = fps[split:]

        choices = random.choices(training, k=n_per_cell_type)
        for i, fp in enumerate(choices):
            shutil.copy(fp,
                fp.replace(f'/all/{cell_type}/', f'/training/{cell_type}/').replace('.jpg', f'_{i}.jpg'))
        for fp in validation[:max_val_per_cell_type]:
            shutil.copy(fp, fp.replace(f'/all/{cell_type}/', f'/validation/{cell_type}/'))
    

class PollockDataset(object):
    def __init__(self, adata, cell_type_key='ClusterName', dataset_type='training',
            image_root_dir=os.path.join(os.getcwd(), 'temp_image'), n_per_cell_type=50,
            max_val_per_cell_type=1000, gene_template=None, cell_type_template=None,
            cell_types=None, batch_size=64, nn_threshold=.05):

        self.image_root_dir = image_root_dir
        self.adata = adata
        self.dataset_type = dataset_type
        self.gene_template = gene_template
        self.cell_type_template = cell_type_template
        self.batch_size = batch_size
        self.cell_types = cell_types
        self.nn_threshold = nn_threshold

        if dataset_type == 'prediction':
            self.prediction_ds = None
            self.prediction_length = None

            self.set_prediction_dataset()
            self.cell_ids = [x.split(os.path.sep)[-1].replace('.jpg', '') 
                    for x in self.prediction_ds.filenames]
##             self.cell_ids = np.asarray(adata.obs.index)
        else:
            self.cell_type_key=cell_type_key
            self.cell_types = sorted(set(self.adata.obs[self.cell_type_key]))
            self.cell_type_to_filesafe = {c:str(uuid.uuid4()) for c in self.cell_types}
            self.filesafe_to_cell_type = {v:k for k, v in self.cell_type_to_filesafe.items()}
            self.filesafe_cell_types = np.asarray([self.cell_type_to_filesafe[c]
                    for c in self.cell_types])
            self.n_per_cell_type=n_per_cell_type
            self.max_val_per_cell_type = max_val_per_cell_type
            self.train_ds = None
            self.val_ds = None
            self.train_length = None
            self.val_length = None

            self.set_training_datasets()
        
    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
##         return parts[-2] == self.cell_types
        return parts[-2] == self.filesafe_cell_types
    
    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
    #     return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
        return img
    
    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000, batch_size=64):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ## remove previous cache if needed
                ## look for cache files
                fps = [fp for fp in pollock_pp.listfiles(os.path.dirname(cache), regex=r'\.tfcache') if cache in fp]
                for fp in fps:
                    os.remove(fp)
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
    
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # Repeat forever
        ds = ds.repeat()
    
        ds = ds.batch(batch_size)
    
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)
    
        return ds

    def get_prediction_dataset(self, fp, batch_size=64):
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        data_gen = image_generator.flow_from_directory(directory=str(fp),
                 batch_size=batch_size,
                 shuffle=False,
                 target_size=(128, 128))
        return data_gen

    def get_training_dataset(self, fp, batch_size=64):
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        data_gen = image_generator.flow_from_directory(directory=str(fp),
                 batch_size=batch_size,
                 shuffle=True,
                 target_size=(128, 128))
        return data_gen
    
    def get_dataset(self, fp, cache=True, batch_size=64):
        list_ds = tf.data.Dataset.list_files(str(fp + '/*/*'))
        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)

##         if isinstance(cache, str):
##             if os.path.exists(cache)
##                 os.path.rm
        return self.prepare_for_training(labeled_ds, cache=cache, batch_size=batch_size)

    def set_image_templates(self, n_genes=100):
        sc.tl.rank_genes_groups(self.adata, self.cell_type_key, method='t-test', n_genes=n_genes)
        self.gene_template, self.cell_type_template = create_block_image_template(self.adata,
                key=self.cell_type_key, nn_threshold=self.nn_threshold)

    def get_cell_image(self, cell_id, show=True):
        X = self.adata[self.adata.obs.index==cell_id].X
        available_genes = set(self.gene_template.flatten())

        if 'sparse' in str(type(X)):
            X = X.toarray()

        genes, idxs = zip(*[(g, i) for i, g in enumerate(self.adata.var.index) if g in available_genes])
        X = X[:, np.asarray(idxs)]
        expression = (X[0] / np.max(X[0])) * 255.
        gene_to_expression = {g:e
                for g, e in zip(genes, expression)}
        gene_img = get_expression_image(gene_to_expression, self.gene_template)
        gene_img = np.moveaxis(np.asarray([gene_img, gene_img, gene_img]), 0, -1)

        if show:
            plt.imshow(gene_img)

        return gene_img

    def write_training_images(self):
##         initialize_directories(self.cell_types, root=self.image_root_dir)
        initialize_directories(self.filesafe_cell_types, root=self.image_root_dir)
        cell_type_to_fps = write_images(self.adata, self.gene_template, self.cell_type_to_filesafe,
                root=self.image_root_dir, cell_type_key=self.cell_type_key)
        setup_training(cell_type_to_fps, n_per_cell_type=self.n_per_cell_type,
                max_val_per_cell_type=self.max_val_per_cell_type)

    def write_prediction_images(self):
        initialize_directories(self.cell_types, root=self.image_root_dir, directories=['all'],
                purpose=self.dataset_type)
        write_images(self.adata, self.gene_template, {},
                root=self.image_root_dir, purpose=self.dataset_type, cell_type_key=None)

    def set_training_datasets(self):
        """"""
        logging.info(f'creating image templates')
        self.set_image_templates()
        logging.info(f'writing training images')
        self.write_training_images()
        logging.info(f'creating training dataset')
        self.train_ds = self.get_dataset(os.path.join(self.image_root_dir, 'training'),
                cache=os.path.join(os.getcwd(), 'training.tfcache'), batch_size=self.batch_size)
##         self.train_ds = self.get_training_dataset(os.path.join(self.image_root_dir, 'training'),
##                 batch_size=self.batch_size)
        logging.info(f'creating validation dataset')
        self.val_ds = self.get_dataset(os.path.join(self.image_root_dir, 'validation'),
                cache=os.path.join(os.getcwd(), 'validation.tfcache'), batch_size=self.batch_size)
##         self.val_ds = self.get_training_dataset(os.path.join(self.image_root_dir, 'validation'),
##                 batch_size=self.batch_size)
        self.train_length = len(
                list(pollock_pp.listfiles(os.path.join(self.image_root_dir, 'training'),
                regex='.jpg$')))
        self.val_length = len(
                list(pollock_pp.listfiles(os.path.join(self.image_root_dir, 'validation'),
                regex=r'.jpg$')))

    def set_prediction_dataset(self):
        self.write_prediction_images()
        self.prediction_ds = self.get_prediction_dataset(os.path.join(self.image_root_dir, 'all'),
                batch_size=self.batch_size)
        self.prediction_length = len(
                list(pollock_pp.listfiles(os.path.join(self.image_root_dir, 'all'),
                regex='.jpg$')))



def load_from_directory(adata, model_filepath, image_root_dir=os.path.join(os.getcwd(), 'prediction'),
        batch_size=64):
    model = tf.keras.models.load_model(os.path.join(model_filepath, MODEL_PATH))
    gene_template = np.load(os.path.join(model_filepath, GENE_TEMPLATE_PATH), allow_pickle=True)
    cell_type_template = np.load(os.path.join(model_filepath, CELL_TYPE_TEMPLATE_PATH), allow_pickle=True)
    cell_types = np.load(os.path.join(model_filepath, CELL_TYPES_PATH), allow_pickle=True)

    prediction_dataset = PollockDataset(adata, dataset_type='prediction',
           image_root_dir=image_root_dir,
           gene_template=gene_template,
           cell_type_template=cell_type_template,
           cell_types=cell_types,
           batch_size=batch_size)

    pollock_model = PollockModel(cell_types, img_width=gene_template.shape[1],
            img_height=gene_template.shape[0], model=model)

    return prediction_dataset, pollock_model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class PollockModel(object):
    def __init__(self, class_names, img_width=128, img_height=128, patience=2, model=None):
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]
        self.history = None

        if model is None:
            self.model = Sequential([
                tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                    input_shape=(img_width, img_height, 3)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(.5),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(len(class_names), activation='softmax')
            ])
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        elif model == 'mobilenet':
            base_model = tf.keras.applications.MobileNetV2(input_shape=(img_width, img_height, 3),
                include_top=False)
            self.model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(.5),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(len(class_names), activation='softmax')
                ])
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        elif model == 'resnet50':
            base_model = tf.keras.applications.ResNet50V2(input_shape=(img_width, img_height, 3),
                include_top=False)
            self.model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(.5),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(len(class_names), activation='softmax')
                ])
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        else:
            self.model = model

        self.class_names = class_names

    def fit(self, pollock_dataset, batch_size=64, epochs=10):
        self.history = self.model.fit(pollock_dataset.train_ds,
            epochs=epochs,
            steps_per_epoch=(pollock_dataset.train_length // batch_size) + 1,
            validation_data=pollock_dataset.val_ds,
            validation_steps=(pollock_dataset.val_length // batch_size) + 1,
            callbacks=self.callbacks)

    def predict(self, pollock_dataset):
        return self.model.predict(pollock_dataset.prediction_ds)

    def save(self, pollock_training_dataset, filepath, X_val=None, y_val=None):
        ## create directory if does not exist
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        model_fp = os.path.join(filepath, MODEL_PATH)
        self.model.save(model_fp) 
        np.save(os.path.join(filepath, GENE_TEMPLATE_PATH),
                pollock_training_dataset.gene_template)
        np.save(os.path.join(filepath, CELL_TYPE_TEMPLATE_PATH),
                pollock_training_dataset.cell_type_template)
        np.save(os.path.join(filepath, CELL_TYPES_PATH),
                np.asarray(pollock_training_dataset.cell_types))

        if X_val is not None and y_val is not None:
            probs = self.model.predict(X_val)
            predictions = np.argmax(probs, axis=1).flatten()
            predicted_labels = [self.class_names[i] for i in predictions]

            cell_type_to_index = {v:k for k, v in enumerate(self.class_names)}
            groundtruth = [cell_type_to_index[cell_type]
                    for cell_type in y_val]

            report = classification_report(groundtruth, predictions,
                   target_names=self.class_names, output_dict=True)

            c_df = pollock_analysis.get_confusion_matrix(predictions,
                    groundtruth, self.class_names,  show=False)

            d = {
                    'validation': {
                        'metrics': report,
                        'probabilities': probs,
                        'prediction_labels': predicted_labels,
                        'groundtruth_labels': y_val,
                        'confusion_matrix': c_df.values,
                        },
                    'history': {k:[float(x) for x in v]
                        for k, v in self.history.history.items()},
                    }
            json.dump(d, open(os.path.join(filepath, MODEL_SUMMARY_PATH), 'w'),
                    cls=NumpyEncoder)







