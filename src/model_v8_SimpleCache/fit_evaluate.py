import keras
from keras import Model
import tensorflow as tf
import numpy as np
import pdb

def fit_evaluate(model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr, logstring, cache_fraction=1, lmbd=1):

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['accuracy'])

    tbCallBack = keras.callbacks.TensorBoard(log_dir=logstring, histogram_freq=0,  
          write_graph=True, write_images=True)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
        min_delta=0, patience=10, verbose=2, mode='auto', restore_best_weights=True)

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test), 
            callbacks = [tbCallBack, early_stopping])

    # print('model.layers')
    # for i in range(len(model.layers)):
    #     print(i, model.layers[i])
    #     print(i, np.array(model.layers[i].get_weights()).shape)
    #     #print(i, np.array(model.layers[i].get_weights()))
    # produces:
    # 0 <keras.layers.convolutional.Conv2D object at 0x1332ee080>
    # 1 <keras.layers.core.Activation object at 0x133477cf8>
    # 2 <keras.layers.convolutional.Conv2D object at 0x133477208>
    # 3 <keras.layers.core.Activation object at 0x1334a5b38>
    # 4 <keras.layers.pooling.MaxPooling2D object at 0x1334b6b38>
    # 5 <keras.layers.core.Dropout object at 0x1334a5f60>
    # 6 <keras.layers.convolutional.Conv2D object at 0x1334a5eb8>
    # 7 <keras.layers.core.Activation object at 0x1334cbc18>
    # 8 <keras.layers.convolutional.Conv2D object at 0x1334cb1d0>
    # 9 <keras.layers.core.Activation object at 0x14020de80>
    # 10 <keras.layers.pooling.MaxPooling2D object at 0x14020dd68>
    # 11 <keras.layers.core.Dropout object at 0x14021e470>
    # 12 <keras.layers.core.Flatten object at 0x14021e048>
    # 13 <keras.layers.core.Dense object at 0x116a54400>
    # 14 <keras.layers.core.Activation object at 0x116a7a8d0>
    # 15 <keras.layers.core.Dropout object at 0x116a7a5c0>
    # 16 <keras.layers.core.Dense object at 0x116a7af98>
    # 17 <keras.layers.core.Activation object at 0x116a8e5f8>
    # 18 <keras.layers.normalization.BatchNormalization object at 0x116a8ec50>
    # 19 <keras.layers.core.Dense object at 0x116a8edd8>
    # 20 <keras.layers.core.Activation object at 0x117096da0>
    
    # SIMPLE CACHE
    num_classes = 10
    mem_layers = [14, 17, 20]

    output_list = []
    for i in range(len(mem_layers)):
        output_list.append(model.layers[mem_layers[i]].output)

    # extra memory model:
    memory_extractor = Model(inputs=model.input, outputs=output_list)

    # randomly order the training data indices and select only those up to the selected percentage: 
    cache_indices = np.random.permutation(np.arange(x_train.shape[0]))[:int(x_train.shape[0]*cache_fraction)]

    x_train_sub = x_train[cache_indices, :, :, :]
    y_train_sub = y_train[cache_indices, :]

    # replace x/y_test with x/y_train_sub should return 100% accuracy
    if True:
        x_test = x_train_sub
        y_test = y_train_sub

    if False:
        x_test = x_train_sub[:-1, :, :, :]
        y_test = y_train_sub[:-1, :]

    print('shape')
    print(x_test.shape)
    print(y_test.shape)

    # Pass subset items through memory to extract the activations for each memory point
    memkeys_list = memory_extractor.predict(x_train_sub)
    if len(mem_layers) == 1: memkeys_list = memkeys_list[None,:]
    mem_keys = np.reshape(memkeys_list[0],(x_train_sub.shape[0],-1))
    for i in range(len(mem_layers)-1):
        mem_keys = np.concatenate((mem_keys, np.reshape(memkeys_list[i+1],(x_train_sub.shape[0],-1))),axis=1)

    # Memory values
    mem_vals = y_train_sub

    # Pass test items through memory to extract the activations for each memory point
    testmem_list = memory_extractor.predict(x_test)
    if len(mem_layers) == 1: testmem_list = testmem_list[None,:] 
    test_mem = np.reshape(testmem_list[0],(x_test.shape[0],-1)) # get first activation
    for i in range(len(mem_layers)-1):
        test_mem = np.concatenate((test_mem, np.reshape(testmem_list[i+1],(x_test.shape[0],-1))),axis=1)

    # Normalize keys and query
    key = mem_keys / np.sqrt(np.tile(np.sum(mem_keys**2, axis=1, keepdims=1), (1, mem_keys.shape[1])))
    query = test_mem / np.sqrt(np.tile(np.sum(test_mem**2, axis=1, keepdims=1), (1, test_mem.shape[1])))

    theta = 0.6
    similarities = np.exp(theta * (query@key.T)) # Eq(1)
    similarities_ = np.repeat(np.expand_dims(similarities, axis=2), num_classes, axis=2)
    mem_vals_ = np.repeat(np.expand_dims(mem_vals, axis=1), x_test.shape[0], axis=1)
    p_mem_enum = np.sum(np.multiply(similarities_, mem_vals_), axis=1, keepdims=False) # Eq(2) enumerator
    p_mem_denom = np.repeat(np.expand_dims(np.sum(similarities, axis=0), axis=1), num_classes, axis=1) #Eq(2) denominator
    # p_mem_enum = np.matmul(similarities, mem_vals) # Eq(2) enumerator
    # p_mem_denom = np.repeat(np.sum(similarities, axis=1, keepdims=True), num_classes, axis=1) #Eq(2) denominator
    p_mem = p_mem_enum / p_mem_denom #Eq(2)

    p_model = model.predict(x_test)

    # lmbd is passed 
    lmbd = 0.5
    p_combined = (1.0-lmbd) * p_model + lmbd * p_mem
    pred_combined = np.argmax(p_combined, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    print('This should return 100% Accuracy when using x_test = x_train_sub:')
    test_acc = np.mean(pred_combined==y_test_int)

    print('Mem. shape:', mem_keys.shape)
    print('Mem. accuracy:', test_acc)

    mem_loss, mem_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('model loss:', mem_loss)
    print('model accuracy:', mem_accuracy)
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    pdb.set_trace()

    metrics_dict = {
        'acc': acc,
        'val_acc': val_acc,
        'loss': loss,
        'val_loss': val_loss,
        'mem_loss': mem_loss,
        'mem_acc': mem_accuracy
    }
    
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    
    return metrics_dict, key
