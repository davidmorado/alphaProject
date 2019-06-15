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
testing = True
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
	lr = [0.0001]


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
				for lr in lrs
					lr = float(lr)
					os.system(F"sbatch template.sh {bw} {kpc} {es} {tp} {lr}")
