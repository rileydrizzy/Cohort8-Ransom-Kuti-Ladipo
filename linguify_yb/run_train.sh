#!/bin/bash
echo "Runing train script"
torchrun --standalone --nproc_per_node=1 src/main.py 
#--epochs 10 --batch 512