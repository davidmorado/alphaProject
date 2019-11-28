#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schestag@uni-hildesheim.de


# ## FOR GPU USE:
# #SBATCH --partition=GPU
# #SBATCH --gres=gpu:1
#source activate test

# ## FOR CPU USE:
# #SBATCH --partition=CPU
# #SBATCH --cpus-per-task=40
# source activate MKL

## FOR TEST USE:
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
source activate student_project

## Run the script
srun python main_eval.py
