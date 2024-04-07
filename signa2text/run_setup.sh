#!/usr/bin/env bash

# Exit immediately if any command exits with a non-zero status
set -e

venv="$1"

if [ "$venv" == "no-venv" ]; then
    echo "Installing without creating a virtual environment."
    pip install --no-cache-dir -r requirements.txt
else
    echo "Installing with virtual environment."
    python -m venv env
    source env/bin/activate
    pip install --no-cache-dir -r requirements.txt
fi
