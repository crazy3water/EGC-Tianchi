import math
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard
import matplotlib.pyplot as plt

def CallBackList(PARA):
    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.4
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,
                                         math.floor((1 + epoch) / epochs_drop))

        if (lrate < 4e-5):
            lrate = 4e-5

        print('Changing learning rate to {}'.format(lrate))
        return lrate

    from keras.callbacks import Callback
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = {'batch':[], 'epoch':[]}
            self.accuracy = {'batch':[], 'epoch':[]}
            self.val_loss = {'batch':[], 'epoch':[]}
            self.val_acc = {'batch':[], 'epoch':[]}

        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))

        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))

        def loss_plot(self, loss_type):
            iters = range(len(self.losses[loss_type]))
            plt.figure()
            # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # val_acc
                plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
                # val_loss
                plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.show()

    CB = []

    if PARA['LR']:
        lrate = LearningRateScheduler(step_decay)
        CB.append(lrate)

    if PARA['tensorboard']:
        tensorboard = TensorBoard(log_dir=PARA['tensorboard_path'])
        CB.append(tensorboard)

    if PARA['CSV']:
        csv_log = CSVLogger(filename=PARA['CSV_path'],
                               separator=',',
                               append=True)
        CB.append(csv_log)

    if PARA['EarlyStop']:
        earlystopper = EarlyStopping(monitor='val_f1', patience=5, verbose=1)
        CB.append(earlystopper)

    if PARA['CheckPiont']:
        checkpointer = ModelCheckpoint(PARA['model_path'],
                                       mode='max',
                                       monitor='val_f1',
                                       verbose=1,
                                       save_best_only=True)
        CB.append(checkpointer)
    if PARA['LossHistory']:
        history = LossHistory()
        CB.append(history)
    return CB