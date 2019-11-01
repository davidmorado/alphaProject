import tensorflow as tf
import keras
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50

from varkeys import Varkeys
from data_loader import get_dataset, percentage_splitter

tp = 1
batch_size = 16
learning_rate=0.001
embedding_dim = 100
n_keys_per_class = 100
bandwidth = 100
epochs = 10

# get training data
x_train, x_val, x_test, y_train, y_val, y_test = get_dataset('cifar10', normalize=True, ratio=0.2)
# subsample training data
x_train, y_train = percentage_splitter(x_train, x_val, y_train, y_val, merging=True, random_selection=False, ratio=tp) 
num_categories = y_train.shape[1]
N,h,w,c = x_train.shape
input_shape=h,w,c

RN50 = ResNet50(input_shape=(32, 32, 3), include_top=False)
for layer in RN50.layers[1:-2]:
    layer.trainable = False
memory = Varkeys(keysize=embedding_dim, n_keys_per_class=n_keys_per_class, num_categories=num_categories, bandwidth=bandwidth)

x = RN50.output
x = GlobalAveragePooling2D()(x)
x = Dense(embedding_dim)(x)
predictions = memory(x)

model = Model(input=RN50.input, output=predictions)

model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
    min_delta=0, patience=10, verbose=2, mode='auto', restore_best_weights=True)

history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks = [early_stopping])
