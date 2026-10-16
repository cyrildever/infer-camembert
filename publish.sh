#!/bin/bash

# Remove former distributions
rm dist/infer*

# Publish library to PyPI

python3 -m build
python3 -m twine upload dist/*
