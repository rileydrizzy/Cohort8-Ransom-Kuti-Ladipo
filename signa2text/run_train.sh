#!/bin/bash

echo "Setting Enviroment Variables"
. ./set_environment_variables.sh

# Display a header with script information
echo "=== Running Train Script ==="
python src/main.py

echo "=== Completed Train Script Run ==="