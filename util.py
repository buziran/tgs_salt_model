import math
import functools
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
    def __init__(self, lr, decay, epochs_decay='10', freeze_once=False):
        self.lr = lr
        self.decay = decay
        if ',' in epochs_decay:
            epochs_decay = epochs_decay.split(',')
            if epochs_decay[-1] == '':
                epochs_decay = epochs_decay[:-1]
            epochs_decay = np.array([int(e) for e in epochs_decay])
        else:
            epochs_decay = int(epochs_decay)
        self.epochs_decay = epochs_decay
        self.freeze_once = freeze_once

    def __call__(self, epoch):
       if self.freeze_once and epoch == 0:
           return 0.0
       if isinstance(self.epochs_decay, np.ndarray):
           lr_cur = self.lr * math.pow(self.decay, np.sum(self.epochs_decay <= epoch+1))
       else:
           lr_cur = self.lr * math.pow(self.decay, math.floor((1+epoch)/self.epochs_decay))
       return lr_cur


class CLRDecay(object):
    def __init__(self, lr, max_lr, epoch_size=10, gamma=0.9994, mode='triangular', freeze_once=False):
        self.lr = lr
        self.max_lr = max_lr
        self.epoch_size = epoch_size
        self.gamma = gamma
        self.mode = mode
        self.freeze_once = freeze_once

    def __call__(self, epoch):
        if self.freeze_once and epoch == 0:
            return 0.0
        lr = self.lr
        max_lr = self.max_lr
        epoch_size = self.epoch_size
        gamma = self.gamma
        cycle = math.floor(1 + epoch / (2 * epoch_size))
        x = abs(epoch / epoch_size - 2 * cycle + 1)
        clr = lr + (max_lr - lr) * max(0, 1 - x)

        if self.mode == 'triangular2':
            clr = clr / pow(2, cycle - 1)
        if self.mode == 'exp_range':
            clr = clr * pow(gamma, epoch)
        return clr


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


def get_metrics(threshold=None):
    if threshold is None:
        return [weighted_mean_iou, weighted_mean_score]
    else:
        _weighted_mean_score = functools.partial(weighted_mean_score, threshold=threshold)
        functools.update_wrapper(_weighted_mean_score, weighted_mean_score)
        return [weighted_mean_iou, _weighted_mean_score]


def get_custom_objects():
    return {'weighted_mean_iou': weighted_mean_iou, 'weighted_mean_score': weighted_mean_score}

def write_summary(model, filename):
    with open(filename, 'w') as f:
        _print = lambda x: print(x, file=f)
        model.summary(print_fn=_print)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

