#!/usr/bin/env bash
#SBATCH --job-name=proto_res4
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dedeoglu@uni-hildesheim.de



# ## FOR GPU USE:
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate projectGPU


## Run the script
srun python cluster_proto_resnet24.py

