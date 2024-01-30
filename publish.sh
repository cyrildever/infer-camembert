#!/bin/bash

# Remove former distributions
rm dist/infer_camembert-*
rm dist/infer-camembert-*

# Publish library to PyPI

python3 -m build
python3 -m twine upload dist/*
