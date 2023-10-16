import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class LossRecord:
    def __init__(self, path):
        self.path = path

    def record(self, loss):
        with open(self.path, 'a') as f:
            f.write(str(loss) + '\n')