#!/usr/bin/env bash
#SBATCH --job-name=RN50VK
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schestag@uni-hildesheim.de

# ## FOR GPU USE:
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
source activate lalala

## Run the script
srun python RNcolab.py
