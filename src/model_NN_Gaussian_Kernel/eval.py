import numpy as np 
import matplotlib.pyplot as plt 
import os 
import sys 
import pickle


result_path = 'gridresults/'
results = [item for item in os.listdir(result_path) if 'BEST' not in item and 'old' not in item and '/' not in item]
BEST_result = 'BEST_update_period=2&BEST_nearest_neighbors=100.pkl'
colors = ['orange', 'yellow', 'green', 'blue', 'red', 'black']
for result, color in zip(results, colors):
    with open(result_path + result, 'rb') as f:
       if color=='yellow':
           opacity=0.5
       else:
           opacity = 1
       hp_dict, acc_train_list, acc_validation_list = pickle.load(f)
       plt.plot(acc_train_list, label=F"updatePeriod={hp_dict['update_period']}, nn={hp_dict['nearest_neighbors']}", color=color, alpha=opacity)
       plt.plot(acc_validation_list, 'k--', color=color)
plt.grid()
plt.legend()
plt.savefig(F'plots/gridsearch.png')
plt.show()
plt.clf() 


with open(result_path + BEST_result, 'rb') as f:
    hp_dict, acc_train_list, acc_validation_list = pickle.load(f)
    print(hp_dict)
    print('final test accuracy: ', acc_validation_list[-1])
    print('final mean test accuracy: ', np.mean(acc_validation_list[-15:]))
    plt.plot(acc_train_list, label=F"updatePeriod={hp_dict['update_period']}, nn={hp_dict['nearest_neighbors']}, test_accuracy: {acc_validation_list[-1]:.4f}", color='blue')
    plt.plot(acc_validation_list, 'k--', color='blue')
    plt.legend()
    plt.grid()
    plt.savefig(F'plots/best.png')
    plt.show()

