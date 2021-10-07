import abc
import numpy as np


class Scaler:
    def __init__(self, axis=None):
        self.axis = axis

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def transform(self, data):
        pass

    @abc.abstractmethod
    def inverse_transform(self, data):
        pass


class StandardScaler(Scaler):
    def __init__(self, axis=None, **kwargs):
        super(StandardScaler, self).__init__(axis)
        self.mean = kwargs.get('mean', None)
        self.std = kwargs.get('std', None)

    def fit(self, data):
        if type(data) != np.ndarray:
            data = np.array(data, dtype=np.float32)

        self.mean = data.mean(axis=self.axis)
        self.std = data.std(axis=self.axis)

    def transform(self, data, axis=None):
        try:
            return (data - self.mean[axis]) / self.std[axis]
        except:
            return (data - self.mean) / self.std

    def inverse_transform(self, data, axis=None):
        try:
            return data * self.std[axis] + self.mean[axis]
        except:
            return data * self.std + self.mean


class MinMaxScaler(Scaler):
    def __init__(self, axis=None, **kwargs):
        super(MinMaxScaler, self).__init__(axis)
        self.max = kwargs.get('max', None)
        self.min = kwargs.get('min', None)

    def fit(self, data):
        if type(data) != np.ndarray:
            data = np.array(data, dtype=np.float32)

        self.max = data.max(axis=self.axis)
        self.min = data.min(axis=self.axis)

    def transform(self, data, axis=None):
        try:
            return (data - self.min[axis]) / (self.max[axis] - self.min[axis])
        except:
            return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data, axis=None):
        try:
            return data * (self.max[axis] - self.min[axis]) + self.min[axis]
        except:
            return data * (self.max - self.min) + self.min
