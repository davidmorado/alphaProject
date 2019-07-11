#!/usr/bin/env bash
<<<<<<< HEAD
#SBATCH --job-name=CNN_VK
#SBATCH --output=logs/test%j.log
#SBATCH --error=errs/test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dedeoglu@uni-hildesheim.de

# ## FOR GPU USE:
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
source activate test

## Run the script
srun python main_test.py $1 $2
=======
#SBATCH --job-name=test_alperen
#SBATCH --output=test%j.log
#SBATCH --error=test%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dedeoglu@uni-hildesheim.de



# ## FOR GPU USE:
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source activate projectGPU


## Run the script
srun python CNN+Varkeys+Keras_alperen.py
>>>>>>> 5e53f16712e3b4397369b1c503510fca513c7120
