import numpy as np 
import matplotlib.pyplot as plt 
import os 
import sys 
import pickle


result_path = 'gridresults/'
results = [item for item in os.listdir(result_path) if 'BEST' not in item and 'old' not in item and '/' not in item]

colors = ['orange', 'yellow', 'green', 'blue', 'red', 'black', 'violet', 'brown', 'gray']
# for result, color in zip(results, colors):
#     with open(result_path + result, 'rb') as f:
#        hp_dict, history = pickle.load(f)
#        acc_train_list, acc_validation_list = history['train']['no_memory']['acc'], history['valid']['no_memory']['acc']
#        acc_train_list = [item[1] for item in acc_train_list] # items are of form (epoch, accuracy)
#        acc_validation_list = [item[1] for item in acc_validation_list]
       
#        plt.plot(acc_train_list, label=F"nodes={hp_dict['nodes_in_extra_layer']}, dropout={hp_dict['dropout_in_extra_layer']}", color=color)
#        plt.plot(acc_validation_list, 'k--', color=color)
# plt.grid()
# plt.legend()
# plt.savefig(F'plots/gridsearch.png')
# plt.show()
# plt.clf() 



result_path = 'evaluation/cifar10/'
results = [item for item in os.listdir(result_path) if 'BEST' in item and 'old' not in item and '/' not in item]
scores = [] # items: (val_accuracy, train_ratio)
for result in results:
    with open(result_path + result, 'rb') as f:
        hp_dict, history = pickle.load(f)
        acc_train_list, acc_validation_list = history['train']['no_memory']['acc'], history['valid']['no_memory']['acc']
        acc_train_list = [item[1] for item in acc_train_list] # items are of form (epoch, accuracy)
        acc_validation_list = [item[1] for item in acc_validation_list]
        score = max(acc_validation_list)
        scores.append((score, hp_dict['tr']))

scores = sorted(scores, key=lambda x: x[1])
scores = [score[0] for score in scores]

print(scores)

plt.plot(scores)
plt.xticks(np.arange(len(scores)), ('10%', '20%', '30%', '50%', '75%', '100%'))
plt.grid()
plt.savefig(F'plots/sample_efficiency.png')
plt.show()

