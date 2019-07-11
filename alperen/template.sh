#!/usr/bin/env bash

#SBATCH --job-name=test_alperen
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dedeoglu@uni-hildesheim.de

# ## FOR GPU USE:
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate projectGPU


## Run the script
srun python CNN+Varkeys+Keras_alperen.py

