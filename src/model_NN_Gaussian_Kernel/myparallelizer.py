
import os
import yaml
from sklearn.model_selection import ParameterGrid
import argparse
import subprocess

# creates folders
folders = ['models', 'gridresults', 'plots', 'tb_logs', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass


with open("config.yml", 'r') as stream:
    try:
        hp_dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

params_grid = ParameterGrid(hp_dict)

for combination in params_grid:
	#subprocess.call(['sbatch template.sh'] + ['--' + str(param) + ' ' + str(value) for param, value in combination.items()]])
	combination = str(combination)
	os.system(F"sbatch template.sh {combination}")
    


