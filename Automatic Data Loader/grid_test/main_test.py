import pickle
res = [1,2,3]
with open('test_results/file_test.pkl', 'wb') as f:
  pickle.dump(res, f)

import numpy as np
import sys
import os
import ast

# Hyperparameters:
hp_dict_str = sys.argv[1]

hp_dict = ast.literal_eval(hp_dict_str)

print('PRINT DICTIONARY', hp_dict_str)


os.makedirs('test_results')
out_results = [1,2,3]
filename = ''
for hyp in hp_dict:
    filename += '_'+str(hp_dict[hyp])

filename = F"'test_results'/{filename}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(out_results, f)


