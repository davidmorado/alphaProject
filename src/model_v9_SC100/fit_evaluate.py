import keras
from keras import Model
import tensorflow as tf
import numpy as np
import pdb

def fit_evaluate(model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr, logstring, theta=0.5, cache_fraction=0.1, lmbd=0.5):

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
    
    # SIMPLE CACHE
    num_classes = 10
    mem_layers = [14, 17, 20]
    mem_layers = [17]
    
    output_list = []
    for i in range(len(mem_layers)):
        output_list.append(model.layers[mem_layers[i]].output)

    # Specify memory
    mem = Model(inputs=model.input, outputs=output_list )

    # subsample data points for memory module (further subsampling from potentially subsampled training data)
    idx = np.random.permutation(np.arange(x_train.shape[0]))[:int(x_train.shape[0]*cache_fraction)]

    x_train_in_mem = x_train[idx, :, :, :]
    y_train_in_mem = y_train[idx, :]

    # assert 100% accuracy for x_test == x_train_in_mem
    if False:
        x_test = x_train_in_mem[:-1, :, :]
        y_test = y_train_in_mem[:-1, :]

    # Memory keys
    memkeys_list = mem.predict(x_train_in_mem)
    mem_keys = np.reshape(memkeys_list[0],(x_train_in_mem.shape[0],-1))
    for i in range(len(mem_layers)-1):
        mem_keys = np.concatenate((mem_keys, np.reshape(memkeys_list[i+1],(x_train_in_mem.shape[0],-1))),axis=1)

    # Memory values
    mem_vals = y_train_in_mem

    # Pass items thru memory
    testmem_list = mem.predict(x_test)
    test_mem = np.reshape(testmem_list[0],(x_test.shape[0],-1))
    for i in range(len(mem_layers)-1):
        test_mem = np.concatenate((test_mem, np.reshape(testmem_list[i+1],(x_test.shape[0],-1))),axis=1)

    # Normalize keys and query
    query = test_mem / np.sqrt( np.tile(np.sum(test_mem**2, axis=1, keepdims=1),(1,test_mem.shape[1])) )
    key = mem_keys / np.sqrt( np.tile(np.sum(mem_keys**2, axis=1, keepdims=1),(1,mem_keys.shape[1])) )

    similarities = np.exp(theta * np.dot(query, key.T) )
    p_mem = np.matmul(similarities, mem_vals)
    p_mem = p_mem / np.repeat(np.sum(p_mem, axis=1, keepdims=True), num_classes, axis=1)

    p_model = model.predict(x_test)

    p_combined = (1.0-lmbd) * p_model + lmbd * p_mem
    pred_combined = np.argmax(p_combined, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    test_acc = np.mean(pred_combined==y_test_int)
    mem_only_acc = np.mean(np.argmax(p_mem, axis=1)==y_test_int)

    print('Mem. shape:', mem_keys.shape)
    print('Mem. accuracy:', mem_only_acc)
    print('Mem. + model accuracy:', test_acc)

    mem_loss, mem_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print('model loss:', mem_loss)
    print('model accuracy:', mem_accuracy)
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    metrics_dict = {
        'model_only_acc_train': acc,
        'model_only_acc_val': val_acc,
        'model_only_loss_train': loss,
        'model_only_loss_val': val_loss,
        'model_loss_test': mem_loss,
        'model_acc_test': mem_accuracy,
        'mem_only_acc_test': mem_only_acc,
        'comb_acc_test': test_acc
    }
    
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    pdb.set_trace()
    
    return metrics_dict, key
