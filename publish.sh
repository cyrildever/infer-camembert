#!/bin/bash

# Remove former distributions
rm dist/py_utls-*
rm dist/py-utls-*

# Publishes library to PyPI

python3 -m build
python3 -m twine upload dist/*
