import os
from keras.callbacks import Callback

def assertfolders(folders = ['models', 'gridresults', 'tb_logs', 'errs', 'logs']):
    # creates folders
    for f in folders:
        try:
            os.makedirs(f)
        except OSError:
            pass

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))