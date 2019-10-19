# embedding sizes
ess = [300, 100, 30, 10]
# percentage of training data used
tps = [1, 0.5, 0.25, 0.125]
# learning rates
lrs = [1e-6, 1e-5, 1e-4, 1e-3]
# fraction of data in memory
data_frac = [1, 0.5, 0.25]

##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = True
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
		for lr in lrs:
			lr = float(lr)
			for df in data_frac:
				df = float(df)
				os.system(F"sbatch template.sh {es} {tp} {lr} {df}")
