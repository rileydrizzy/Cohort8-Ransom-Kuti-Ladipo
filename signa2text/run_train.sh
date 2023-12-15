#!/bin/bash

# Display a header with script information
echo "=== Running Train Script ==="

torchrun --standalone --nproc_per_node=1 src/main.py --model_name test_model --epoch 2 
#torchrun --standalone --nproc_per_node=1 src/main.py 
#--epochs 10 --batch 512