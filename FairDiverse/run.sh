#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --out=logs/%j.out

model=$1
if [ -z "$model" ]; then
    echo "Usage: $0 file"
    exit 1

# File does not exist
elif [ ! -f "$model" ]; then
    echo "Model file not found: $model"
    exit 1
fi

echo "----------------------------------------"
echo "Running $model with name: $2"
echo "----------------------------------------"

source activate recsys || conda activate recsys
srun python3 $model