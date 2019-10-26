#!/usr/bin/env bash
#SBATCH --job-name=CNNVK
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=schestag@uni-hildesheim.de

# ## FOR GPU USE:
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
source activate GPU

## Run the script
srun python main.py $1 $2 $3
