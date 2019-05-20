#!/usr/bin/env bash
#SBATCH --job-name=1
#SBATCH --output=keysize=20&nkeys=5.log
#SBATCH --error=keysize=20&nkeys=5.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kurzendo@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate CPU

## Run the script
srun /home/kurzendo/miniconda3/envs/CPU/bin/python /home/kurzendo/srp/varkeys/resnet_cache_keysize20nkeys5.py
