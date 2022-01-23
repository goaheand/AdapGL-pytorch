import numpy as np


def get_mae(y_pred, y_true):
    non_zero_pos = y_true != 0
    # non_zero_pos = range(y_pred.shape[0])
    return np.fabs((y_true[non_zero_pos] - y_pred[non_zero_pos])).mean()


def get_rmse(y_pred, y_true):
    non_zero_pos = y_true != 0
    # non_zero_pos = range(y_pred.shape[0])
    return np.sqrt(np.square(y_true[non_zero_pos] - y_pred[non_zero_pos]).mean())


def get_mape(y_pred, y_true):
    non_zero_pos = (np.fabs(y_true) > 0.5)
    return np.fabs((y_true[non_zero_pos] - y_pred[non_zero_pos]) / y_true[non_zero_pos]).mean()
