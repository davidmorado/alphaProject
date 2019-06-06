#!/usr/bin/env bash
#SBATCH --job-name=CNN_VK_v2
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schestag@uni-hildesheim.de


## FOR GPU USE:
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate test

## Run the script
srun python model_v2/main.py
