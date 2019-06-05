import matplotlib.pyplot as plt
import numpy as np 
import pickle 


# load data:
with open("data/test_scores_memory", 'rb') as f:
    test_scores_memory = pickle.load(f)

with open("data/test_scores_no_memory", 'rb') as f:
    test_scores_no_memory = pickle.load(f)

training_ratios = [0.1, 0.2, 0.3]

# plot model performance with memory
test_loss_memory, test_acc_memory = zip(*test_scores_memory)
test_loss_no_memory, test_acc_no_memory = zip(*test_scores_no_memory)

plt.plot(training_ratios, test_acc_memory, label='memory')#
plt.plot(training_ratios, test_acc_no_memory, label='no memory')
plt.title('Comparison on different training ratios')
plt.ylabel('test accuracy')
plt.xlabel('training ratio')
plt.xticks(training_ratios)
plt.legend()
#plt.savefig('plots/acc_lr={}.png'.format(LR) )
plt.show()
plt.clf()

plt.plot(training_ratios, test_loss_memory, label='memory')#
plt.plot(training_ratios, test_loss_no_memory, label='no memory')
plt.title('Comparison on different training ratios')
plt.ylabel('test loss')
plt.xlabel('training ratio')
plt.xticks(training_ratios)
plt.legend()
#plt.savefig('plots/loss_lr={}.png'.format(LR) )
plt.show()
plt.clf()