from setuptools import setup, find_packages
from setuptools.command.install import install
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    # $ pip install pollock
    name='dinglab-pollock',
    version='0.0.5',
    description='A tool for single cell classification and characterization.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ding-lab/pollock',
    author='Ding Lab',
    author_email='estorrs@wustl.edu',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='single cell classification expression machine learning deep learning',  # Optional
    #packages=find_packages(exclude=['tests']),
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'scanpy>=1.4.4',
        'pandas>=1.0.0',
        'seaborn>=0.10.0',
        'scipy>=1.4.1',
        'scikit-learn>=0.22.1',
        'tensorflow==2.1.0',
        'jupyter',
        'umap-learn>=0.3.10',
        'loompy>=2.0.17',
        'matplotlib>=3.2.1',
        ],
    include_package_data = True,

    entry_points={ 
        'console_scripts': [
            'pollock=pollock.pollock:main',
            'pollock-setup=pollock.pollock_setup:main',
        ],
   },
)
