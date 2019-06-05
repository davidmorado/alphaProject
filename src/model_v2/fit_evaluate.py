import keras
import tensorflow as tf

def fit_evaluate(model, x_train, y_train, x_test,  y_test, batch_size, epochs, lr):

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.rmsprop(lr=lr, decay=1e-6),
                metrics=['accuracy'])

    tbCallBack = keras.callbacks.TensorBoard(log_dir='tb_logs/', histogram_freq=0,  
          write_graph=True, write_images=True)

    history = model.fit(x_train, y_train,
            batch_size=  batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test), 
            callbacks = [tbCallBack])

    memory = model.layers[-1].get_memory()
    memory = tf.Session().run(memory)
    
    val_acc = history.history['val_acc']
    acc = history.history['acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return val_acc, acc, loss, val_loss, scores, memory
