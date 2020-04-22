import argparse
import logging
import subprocess

#from pollock import install_seurat

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('command', type=str,
        help='Setup command to run. Possible commands are: from_seurat')

args = parser.parse_args()

def setup_seurat():
    """Installs packages for seurat conversion.
    Note: These dont always play nicely with CUDA
    """
    logging.info('installing packages for seurat conversion')
#    subprocess.check_output(('conda', 'install', '-y', 'r-base'))
    logging.info('installing packages from bioconda: loomR, scater, seurat')
##     subprocess.check_output(('conda', 'install', '-y', '-c', 'bioconda', 'r-loom'))
##     subprocess.check_output(('conda', 'install', '-y', '-c', 'bioconda', 'bioconductor-scater', 'r-loom'))
##     subprocess.check_output(('conda', 'install', '-y', '-c', 'bioconda', 'r-seurat==3.0.2',
##             'bioconductor-scater', 'r-loom'))
    subprocess.check_output(('conda', 'install', '-y', '-c', 'bioconda', 'r-seurat==3.0.2',
        'bioconductor-scater==1.14.0', 'r-loom==0.2.0.2'))

##     install_seurat()

    logging.info('finished installation')

def main():
    if args.command == 'from_seurat':
        setup_seurat()
    else:
        raise RuntimeError(f'{args.command} is not a valid command')
