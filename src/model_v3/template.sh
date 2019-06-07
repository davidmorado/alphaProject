#!/usr/bin/env bash
#SBATCH --job-name=CNN_VK
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=ALL
##SBATCH --mail-user=schestag@uni-hildesheim.de
#SBATCH --mail-user=fschesta@mail.sfsu.edu

# ## FOR GPU USE:
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate test

## Run the script
srun python main.py $1 $2 $3 $4 $5 $6
