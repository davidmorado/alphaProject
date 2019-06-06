# learning rates
lrs = [1e-4, 1e-3, 1e-2, 1e-1]
# bandwith sizes
bws = [1e-2, 1e-1, 1, 1e1, 1e2]
# number of keys per class
kpcs = [1, 1e1, 1e2, 1e3]
# embedding sizes
ess = [30, 300, 3000]
# percentage of training data used
tps = [1, 0.5, 0.25, 0.125]

##########
lrs = [1e-2]
# bandwith sizes
bws = [1]
# number of keys per class
kpcs = [1e2]
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

for lr in lrs:
	for bw in bws:
		for kpc in kpcs:
			for es in ess:
				for tp in tps:
					#print(F"sbatch template.sh {lr} {bw} {kpc} {es} {tp}")
					os.system(F"sbatch template.sh '{lr}' '{bw}' '{kpc}' '{es}' '{tp}'")
