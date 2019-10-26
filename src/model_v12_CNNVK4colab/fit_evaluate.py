import keras
import tensorflow as tf

def fit_evaluate(model, x_train, y_train, x_val, y_val, x_test,  y_test, batch_size, epochs, lr, logstring):

    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(
            lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        metrics=['accuracy'])

    tbCallBack = keras.callbacks.TensorBoard(log_dir=logstring, histogram_freq=0,
          write_graph=True, write_images=True)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
        min_delta=0, patience=10, verbose=2, mode='auto', restore_best_weights=True)

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_val, y_val),
            callbacks = [tbCallBack, early_stopping])

    # memory = model.get_weights()[-1]

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
    
    return metrics_dict, scores # memory
