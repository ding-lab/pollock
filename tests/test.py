import pytest
import os

import subprocess

def test_basic_cell_prediction():
    tool_args = ('pollock', '--output', 'output.tsv', 'tests/data/mini_expression_matrix.tsv')
    subprocess.check_output(tool_args)

    assert True

def test_cell_prediction_with_confidence():
    tool_args = ('pollock', '--min-confidence-level', '.99', '--output', 'output.tsv',
            'tests/data/mini_expression_matrix.tsv')
    subprocess.check_output(tool_args)

    assert True
    


