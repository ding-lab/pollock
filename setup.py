from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    # $ pip install pollock
    name='pollock',
    version='0.0.2',
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
    packages=['pollock'],
    python_requires='>=3.6',
    install_requires=[
        'pandas>=1.0.0',
        'anndata>=0.7.1',
        'opencv-python>=4.2.0.32',
        'seaborn>=0.10.0',
        'scipy>=1.4.1',
        'scanpy==1.4.5.post3',
        'Pillow>=7.0.0',
        'scikit-learn>=0.22.1',
        'tensorflow==2.0.0',
        'jupyter',
        'tf-explain==0.2.0',
        'umap-learn>=0.3.10'
        ],

##     entry_points={  # Optional
##         'console_scripts': [
##             'pollock=pollock.pollock:main',
##         ],
##    },
)
