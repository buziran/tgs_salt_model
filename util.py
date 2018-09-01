import math

from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
import numpy as np

from metrics import weighted_mean_score, weighted_mean_iou


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


class StepDecay(object):
    def __init__(self, lr, decay, epochs_decay=10):
        self.lr = lr
        self.decay = decay
        self.epochs_decay = epochs_decay

    def __call__(self, epoch):
       lr_cur = self.lr * math.pow(self.decay, math.floor((1+epoch)/self.epochs_decay))
       return lr_cur


class MyTensorBoard(TensorBoard):
    def __init__(self, log_dir, model):
        super().__init__(log_dir=log_dir)
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def load_npz(path_pred):
    npzfile = np.load(path_pred)
    return npzfile['arr_0']


def get_metrics():
    return [weighted_mean_iou, weighted_mean_score]


def get_custom_objects():
    return {'weighted_mean_iou': weighted_mean_iou, 'weighted_mean_score': weighted_mean_score}
