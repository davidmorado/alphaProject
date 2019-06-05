#!/usr/bin/env bash
#SBATCH --job-name=cache
#SBATCH --output=logs/Array_.%A_%a.log
#SBATCH --error=err_logs/Array_.%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kurzendo@uni-hildesheim.de
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

echo "$SLURM_ARRAY_TASK_ID"

source activate GPU

## Run the script


# source for job arrays
# https://stackoverflow.com/questions/41900600/slurm-sbatch-job-array-for-the-same-script-but-with-different-input-arguments-ru
# https://help.rc.ufl.edu/doc/SLURM_Job_Arrays
# https://crc.ku.edu/hpc/slurm/how-to/arrays
# gg http://scicomp.aalto.fi/triton/tut/array.html

# for learningrate in 0.00001 0.0001 0.001 0.01
case $SLURM_ARRAY_TASK_ID in
    0) ARGS="0.000001" ;;
    1) ARGS="0.00001" ;;
    2) ARGS="0.0001" ;;
    3) ARGS="0.001" ;;
    4) ARGS="0.01" ;;
    5) ARGS="0.00005" ;;

esac

KEY_SIZE=20  # keysize (= embedding size)
NUM_KEYS=100 # number of keys per class
LR=$ARGS # learning rate
BANDWIDTH=10000 # bandwith parameter
MEMORY=1 # whether to use memory or not


srun /home/kurzendo/miniconda3/envs/GPU/bin/python /home/kurzendo/srp/varkeys/src/CNN+TrainingRatio/cnn_cache.py $KEY_SIZE $NUM_KEYS $LR $BANDWIDTH $MEMORY 

















# for learningrate in 0.00001 0.0001 0.001 0.01
# do
# 	##SBATCH --output=varkeys_LR$LR.log
#     ##SBATCH --error=varkeys_LR$LR.err
#     KEY_SIZE=20  # keysize (= embedding size)
#     NUM_KEYS=100 # number of keys per class
#     LR=$learningrate # learning rate
#     BANDWIDTH=10000 # bandwith parameter
#     MEMORY=1 # whether to use memory or not
#     echo KEY_SIZE $KEY_SIZE
#     echo NUM_KEYS $NUM_KEYS
#     echo LR $LR
#     echo BANDWIDTH $BANDWIDTH
#     echo MEMORY $MEMORY
#     # run with memory cache
#     srun /home/kurzendo/miniconda3/envs/GPU/bin/python /home/kurzendo/srp/varkeys/src/resnet_cache.py $KEY_SIZE $NUM_KEYS $LR $BANDWIDTH $MEMORY >> varkeys_LR$LR.log

#     # run without memory cache
#     ##SBATCH --output=noCachevarkeys_LR$LR.log
#     ##SBATCH --error=noCachevarkeys_LR$LR.err
#     srun /home/kurzendo/miniconda3/envs/GPU/bin/python /home/kurzendo/srp/varkeys/src/resnet_cache.py $LR  $MEMORY >> noCachevarkeys_LR$LR.log

# done

