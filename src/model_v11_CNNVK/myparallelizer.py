import os
from utils import assertfolders
assertfolders()

# bandwith sizes
bws = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
# number of keys per class
kpcs = [1, 1e1, 1e2, 1e3]
# percentage of training data used
tps = [1, 0.5, 0.25, 0.125, 0.0625]

##########
# TEST MODE
# don't forget to set partition=TEST in template.sh
testing = True
if testing:
	# bandwith sizes
	bws = [10]
	# number of keys per class
	kpcs = [1e3]
	# percentage of training data used
	tps = [0.125]

for bw in bws:
	bw = float(bw)
	for kpc in kpcs:
		kpc = int(kpc)
		for tp in tps:
			tp = float(tp)
			os.system(F"sbatch template.sh {bw} {kpc} {tp}")
