import os
import pathlib
import re
import subprocess

import logging

CONVERT_SEURAT_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
        'convert_seurat.R')
TO_SEURAT_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
        'to_seurat.R')
INSTALL_SEURAT_SCRIPT = os.path.join(pathlib.Path(__file__).parent.absolute(),
        'install_seurat.R')

def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)


def write_loom(rds_fp, loom_fp):
    """Convert seurat rds file to loom file"""
    output = subprocess.check_output(
            ('Rscript', CONVERT_SEURAT_SCRIPT, rds_fp, loom_fp))
    logging.info(output)

def save_rds(loom_fp, rds_fp):
    """"""
    output = subprocess.check_output(
            ('Rscript', TO_SEURAT_SCRIPT, loom_fp, rds_fp))
    logging.info(output)

def install_seurat():
    """"""
    output = subprocess.check_output(
            ('Rscript', INSTALL_SEURAT_SCRIPT))
    logging.info(output)
