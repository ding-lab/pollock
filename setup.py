from setuptools import setup, find_packages
from setuptools.command.install import install
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    # $ pip install pollock
    name='dinglab-pollock',
    version='0.0.15',
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
        'Programming Language :: Python :: 3.8',
    ],
    keywords='single cell classification expression machine learning deep learning',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        ],
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'pollock=pollock.pollock:main',
        ],
    },
)
