#!/usr/bin/env bash
#SBATCH --job-name=CNN_VK_K
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schestag@uni-hildesheim.de


# ## FOR GPU USE:
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
source activate test

## Run the script
# #SBATCH --export=NONE
#source /home/schestag/miniconda3/bin/activate test
#srun /home/schestag/miniconda3/envs/test/bin/python test.py
srun python MANN_vary_numberofkeys.py
