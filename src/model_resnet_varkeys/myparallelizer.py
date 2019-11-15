
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

parser = argparse.ArgumentParser()
parser.add_argument("--platform", type=str)
args = parser.parse_args()
platform = args.platform




with open("config.yml", 'r') as stream:
    try:
        hp_dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

params_grid = ParameterGrid(hp_dict)

for i, combination in enumerate(params_grid):
    #subprocess.call(['sbatch template.sh'] + ['--' + str(param) + ' ' + str(value) for param, value in combination.items()]])
    combination = F"{combination}".replace(' ', '')#.replace('\'', '')
    print(combination)
    if platform == 'slurm':
	    os.system(F"sbatch template.sh {combination}")
    elif platform == 'local':
        print(i)
        os.system(F'template_python.sh "{combination}" {i}')
    else:
        raise('Platform (--platform) must be specified as one of (slurm, local)')


