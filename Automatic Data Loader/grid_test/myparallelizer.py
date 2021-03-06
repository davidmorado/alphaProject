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
	combination = str(combination)
	os.system(F"sbatch template.sh {combination}")


""" 
# learning rates
lrs = [10**(-x) for x in range(0, 5)]
# bandwith sizes
bws = [1e-2, 1e-1, 1, 1e1, 1e2]
# number of keys per class
kpcs = [1, 1e1, 1e2, 1e3]
# embedding sizes
ess = [300, 100, 30, 10]
# percentage of training data used
tps = [1, 0.5, 0.25, 0.125]

##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = False
if testing:
	# bandwith sizes
	bws = [10]
	# number of keys per class
	kpcs = [1e3]
	# embedding sizes
	ess = [300]
	# percentage of training data used
	tps = [0.125]
	# learning rate
	lrs = [0.0001]

# 10 1000 300 0.01 0.0001
import os

# creates folders
folders = ['models', 'gridresults', 'tb_logs', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass

for bw in bws:
	bw = float(bw)
	for kpc in kpcs:
		kpc = int(kpc)
		for es in ess:
			es = int(es)
			for tp in tps:
				tp = float(tp)
				for lr in lrs:
					lr = float(lr)
					os.system(F"sbatch template.sh {bw} {kpc} {es} {tp} {lr}")
 """