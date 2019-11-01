import keras
import tensorflow as tf 
import sys 
import numpy as np

#from keras.applications.resnet50 import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
#from keras.applications.resnet_v2 import ResNet101V2
#from keras.applications.resnet_v2 import ResNet152V2

from varkeys import Varkeys
from data_loader import get_dataset, percentage_splitter
from utils import assertfolders, TestCallback
assertfolders()

# Hyperparameters:
batch_size = 64
epochs = 500
embedding_dim = 100
learning_rate = 0.001

# Hyperparameters:
hp_dict = {
    'bandwidth': float(sys.argv[1]),
    'n_keys_per_class': int(sys.argv[2]),
    'train_percentage': float(sys.argv[3]),
}

bandwidth = hp_dict['bandwidth']
n_keys_per_class = hp_dict['n_keys_per_class']
tp = hp_dict['train_percentage']

# get training data
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar10', normalize=True, ratio=0.2)
# subsample training data
x_train, y_train = percentage_splitter(x_train, x_val, y_train, y_val, merging=True, random_selection=False, ratio=tp) 
num_categories = y_train.shape[1]
N,h,w,c = x_train.shape
input_shape=h,w,c

RN = ResNet50V2(weights='imagenet',include_top=False, input_shape=input_shape, classes=num_categories)
x = RN.output
vk_layer = Varkeys(keysize=embedding_dim, n_keys_per_class=n_keys_per_class, num_categories=num_categories, bandwidth=bandwidth))
x = Varkeys(x)



modelpath = F"CNNVK_bw={hp_dict['bandwidth']}_kpc={hp_dict['n_keys_per_class']}_tp={hp_dict['train_percentage']}"

metrics_dict, scores = fit_evaluate(model, x_train, y_train, x_val, y_val, x_test, y_test, batch_size, epochs, lr=learning_rate, logstring=F'tb_logs/{modelpath}')

out_results = (hp_dict, metrics_dict, scores)
filename = F"gridresults/{modelpath}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(out_results, f)










x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a logistic layer -- 10 classes for CIFAR10
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



# model.compile(
#     optimizer='Adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# history = model.fit(x_train, y_train, validation_data=(x_test, y_test),  epochs=1)
# model.evaluate(x_test, y_test)

# np.save('accuracy', np.array(history.history['acc']))
# np.save('val_accuracy', np.array(history.history['val_acc']))

# plt.plot(history.history['acc'])#
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')