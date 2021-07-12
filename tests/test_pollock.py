import os
import subprocess
import shutil

import pandas as pd
import pytest

from pollock.models.model import PollockDataset, PollockModel, predict_from_anndata

MODULE_FILEPATH = './test_module'
SEURAT_RDS_FILEPATH = './data/dummy.rds'
SEURAT_SMALL_RDS_FILEPATH = './data/dummy_small.rds'
SCANPY_H5AD_FILEPATH = './data/dummy.h5ad'
SCANPY_SMALL_H5AD_FILEPATH = './data/dummy_small.h5ad'
OUTPUT_PREFIX = './output'

EXPECTED_LENGTH = 2700


def clean_up_test_files():
    if os.path.exists(MODULE_FILEPATH):
        shutil.rmtree(MODULE_FILEPATH)
    for fp in [f'{OUTPUT_PREFIX}.rds',
            f'{OUTPUT_PREFIX}.txt', f'{OUTPUT_PREFIX}.h5ad']:
        if os.path.exists(fp):
            os.remove(fp)


def test_train_from_seurat_with_default_activeident():
    command = ('pollock', 'train', 'from_seurat',
            '--seurat-rds-filepath', SEURAT_RDS_FILEPATH,
            '--module-filepath', MODULE_FILEPATH)
    subprocess.check_output(command)

    assert os.path.exists(MODULE_FILEPATH)


def test_train_from_seurat_with_specified_cell_type():
    command = ('pollock', 'train', 'from_seurat',
            '--seurat-rds-filepath', SEURAT_RDS_FILEPATH,
            '--module-filepath', MODULE_FILEPATH,
            '--cell-type-key', 'cell_type')
    subprocess.check_output(command)

    assert os.path.exists(MODULE_FILEPATH)


def test_predict_from_seurat_with_rds_output():
    command = ('pollock', 'predict', 'from_seurat',
            '--seurat-rds-filepath', SEURAT_RDS_FILEPATH,
            '--module-filepath', MODULE_FILEPATH,
            '--output-prefix', OUTPUT_PREFIX)
    subprocess.check_output(command)

    out_fp = f'{OUTPUT_PREFIX}.rds'
    assert os.path.exists(out_fp)


def test_predict_from_seurat_with_txt_output():
    command = ('pollock', 'predict', 'from_seurat',
            '--seurat-rds-filepath', SEURAT_RDS_FILEPATH,
            '--module-filepath', MODULE_FILEPATH,
            '--output-prefix', OUTPUT_PREFIX,
            '--txt-output')
    subprocess.check_output(command)

    out_fp = f'{OUTPUT_PREFIX}.txt'
    df = pd.read_csv(out_fp, sep='\t')
    assert df.shape[0] == EXPECTED_LENGTH


def test_explain_from_seurat():
    command = ('pollock', 'explain', 'from_seurat',
            '--explain-filepath', SEURAT_SMALL_RDS_FILEPATH,
            '--background-filepath', SEURAT_SMALL_RDS_FILEPATH,
            '--module-filepath', MODULE_FILEPATH,
            '--output-prefix', OUTPUT_PREFIX,
            '--background-sample-size', '10')
    subprocess.check_output(command)

    out_fp = f'{OUTPUT_PREFIX}.txt'
    assert os.path.exists(out_fp)

    df = pd.read_csv(out_fp, sep='\t')
    assert df.shape[0] == 10
    assert df.shape[1] >= 1000


def test_train_from_scanpy_with_default_leiden():
    command = ('pollock', 'train', 'from_scanpy',
            '--scanpy-h5ad-filepath', SCANPY_H5AD_FILEPATH,
            '--module-filepath', MODULE_FILEPATH)
    subprocess.check_output(command)

    assert os.path.exists(MODULE_FILEPATH)


def test_train_from_scanpy_with_specified_cell_type():
    command = ('pollock', 'train', 'from_scanpy',
            '--scanpy-h5ad-filepath', SCANPY_H5AD_FILEPATH,
            '--module-filepath', MODULE_FILEPATH,
            '--cell-type-key', 'cell_type')
    subprocess.check_output(command)

    assert os.path.exists(MODULE_FILEPATH)


def test_predict_from_scanpy_with_h5ad_output():
    command = ('pollock', 'predict', 'from_scanpy',
            '--scanpy-h5ad-filepath', SCANPY_H5AD_FILEPATH,
            '--module-filepath', MODULE_FILEPATH,
            '--output-prefix', OUTPUT_PREFIX)
    subprocess.check_output(command)

    out_fp = f'{OUTPUT_PREFIX}.h5ad'
    assert os.path.exists(out_fp)


def test_predict_from_scanpy_with_txt_output():
    command = ('pollock', 'predict', 'from_scanpy',
            '--scanpy-h5ad-filepath', SCANPY_H5AD_FILEPATH,
            '--module-filepath', MODULE_FILEPATH,
            '--output-prefix', OUTPUT_PREFIX,
            '--txt-output')
    subprocess.check_output(command)

    out_fp = f'{OUTPUT_PREFIX}.txt'
    assert os.path.exists(out_fp)


def test_explain_from_scanpy():
    command = ('pollock', 'explain', 'from_scanpy',
            '--explain-filepath', SCANPY_SMALL_H5AD_FILEPATH,
            '--background-filepath', SCANPY_SMALL_H5AD_FILEPATH,
            '--module-filepath', MODULE_FILEPATH,
            '--output-prefix', OUTPUT_PREFIX,
            '--predicted-key', 'cell_type',
            '--background-sample-size', '10')
    subprocess.check_output(command)

    out_fp = f'{OUTPUT_PREFIX}.txt'
    assert os.path.exists(out_fp)

    df = pd.read_csv(out_fp, sep='\t')
    assert df.shape[0] == 10
    assert df.shape[1] >= 1000


def test_clean_files():
    clean_up_test_files()
