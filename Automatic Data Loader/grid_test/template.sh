#!/usr/bin/env bash
#SBATCH --job-name=CNN_VK
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=obandod@uni-hildesheim.de

# ## FOR GPU USE:
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
source activate test

## Run the script
srun python main_test.py $1 $2
