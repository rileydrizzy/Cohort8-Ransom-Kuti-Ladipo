#!/bin/bash

curl -sSL https://install.python-poetry.org | python3 -
poetry install --no-root
echo "Environment setup complete"
