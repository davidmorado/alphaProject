#!/usr/bin/env bash
#SBATCH --job-name=CNN_noVK
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kurzendo@uni-hildesheim.de

# ## FOR GPU USE:
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate GPU

## Run the script
srun python main.py $1
