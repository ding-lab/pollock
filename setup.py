from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    # $ pip install pollock
    name='pollock',
    version='0.0.1',
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
##         'joblib>=0.13.2',
##         'matplotlib>=3.1.1',
##         'numpy>=1.17.0',
##         'pandas>=0.25.0',
##         'pytest>=5.0.1',
##         'seaborn>=0.9.0',
##         'scipy>=1.3.0',
##         'scikit-learn>=0.21.3',
##         'tensorflow==2.0.0-beta1'
        ],

##     entry_points={  # Optional
##         'console_scripts': [
##             'pollock=pollock.pollock:main',
##         ],
##    },
)
