# learning rates
lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# bandwith sizes
bws = []
# number of keys per class
kpcs = []
# embedding sizes
ess = []
# percentage of training data used
tps = []

import os
path = '.'

for lr in lrs:
	for bw in bws:
		for kpc in kpcs:
			for es in ess:
				for tp in tps:
					os.system(F"sbatch {path}/sbatch.sh {lr} {bw} {kpc} {es} {tp}")