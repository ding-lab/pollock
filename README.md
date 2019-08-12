# Pollock

![Image of Pollock](https://github.com/ding-lab/pollock/blob/master/images/pollock.png)

A tool for single cell classification

### Installation
##### Requirements
* OS:
  * macOS 10.12.6 (Sierra) or later
  * Ubuntu 16.04 or later
  * Windows 7 or later (maybe)
  
* Python3.6 or later

* Note: Unfortunately the only Linux distribution Tensorflow is guaranteed to work on is Ubuntu 16.04 or later. Ding Lab peeps this means that pollock will not run on Katmai or Denali (CentOS) unless you are running it in a docker container.

##### To install
```bash
git clone https://github.com/ding-lab/pollock.git
pip install -e pollock
```

### Usage
```bash
pollock [-h] [--min-confidence-level MIN_CONFIDENCE_LEVEL]
        [--output OUTPUT]
        expression_matrix
```

For usage examples see [here](https://github.com/ding-lab/pollock/blob/master/tests/test.py)

##### Arguments

expression_matrix
  * Tab-seperated single cell expression matrix. Expression values must be raw counts. Rows are Ensembl gene ids, columns are cell id. For an example of how expression matrix must be formatted see [here](https://github.com/ding-lab/pollock/blob/master/tests/data/mini_expression_matrix.tsv).
  
--min-confidence-level
  * If a prediction is below this confidence level, the cell in question will be labeled 'unknown'. Confidence value is a float between 0.0-1.0. Default is 0.0. 
  
--output
  * Location to write output file.
  
### Outputs

Output is a .tsv file with the following format: cell_id<tab>cell_prediction
  
Possible cell type predictions:
* melanoma, plasma, macromononeutro, malignantplasma, tcell, dc, bcell, ductal, stroma, fibroblast, endothelial, mast, acinar

### Tests

To run tests

```bash
pytest tests/test.py -vv
```

