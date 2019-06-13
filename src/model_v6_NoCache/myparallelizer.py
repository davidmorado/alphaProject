# learning rates
# not applicable since ADADELTA DEFAULT LEARN RATE (lr=1) IS USED
# lrs = [1e-6, 1e-5, 1e-4, 1e-3]
# bandwith sizes
# bws = [1e-2, 1e-1, 1, 1e1, 1e2]
# number of keys per class
# kpcs = [1, 1e1, 1e2, 1e3]

# HPs above not applicable since the model is without memory
# embedding sizes
ess = [300, 100, 30, 10]
# percentage of training data used
tps = [1, 0.5, 0.25, 0.125]

##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = False
if testing:
	# embedding sizes
	ess = [300]
	# percentage of training data used
	tps = [0.125]


import os

# creates folders
folders = ['models', 'gridresults', 'tb_logs', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass

for es in ess:
	es = int(es)
	for tp in tps:
		tp = float(tp)
		os.system(F"sbatch template.sh {es} {tp}")
