from keras.callbacks import Callback


class ValLossThresholdStopping(Callback):
    def __init__(self,
                 min_loss=0.015,
                 min_val_loss=0.015):
        super(ValLossThresholdStopping, self).__init__()

        self.monitor_loss = 'loss'
        self.monitor_val_loss = 'val_loss'
        self.min_loss = min_loss
        self.min_val_loss = min_val_loss

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        current_val_loss = logs.get(self.monitor_val_loss)
        if current_loss <= self.min_loss or current_val_loss < self.min_val_loss:
            print('\nStop training when loss is %f and val loss is %f' % (current_loss, current_val_loss))
            self.model.stop_training = True
