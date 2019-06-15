import keras
import tensorflow as tf

def fit_evaluate(model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr, logstring):

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.keras.optimizers.Adam(
                    lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))

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

    # get varkeys in the final layers
    #memory = model.layers[-1].get_memory()
    #memory = tf.Session().run(memory)
    memory = model.get_weights()[-1]
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    metrics_dict = {
        'acc': acc,
        'val_acc': val_acc,
        'loss': loss,
        'val_loss': val_loss
    }
    
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return metrics_dict, memory
