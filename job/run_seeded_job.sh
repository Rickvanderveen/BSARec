#!/bin/bash

seeds=(42 43 44 45 46)

for seed in "${seeds[@]}"; do
  sbatch "$1" "$seed"
done
