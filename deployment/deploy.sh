# build package
python setup.py sdist bdist_wheel

# upload to pypi
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# before building conda transfer sha256 to meta.yaml
# build conda
conda-build dinglab-pollock

conda convert --platform all <package_path>

anaconda upload <package_path>
