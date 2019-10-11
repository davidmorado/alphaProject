import yaml
from sklearn.model_selection import ParameterGrid
import os

with open("config.yml", 'r') as stream:
    try:
        hp_dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

params_grid = ParameterGrid(hp_dict)

for combination in params_grid:
    print(type(combination))
    combination = str(combination)
    
    #print(F"sbatch template.sh {combination}")
	#os.system(F"sbatch template.sh {combination}")
