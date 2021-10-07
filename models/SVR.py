import numpy as np
from sklearn.svm import SVR


class SVRModel:
    def __init__(self):
        self.svr_list = None

    def fit(self, x, y, *args, **kwargs):
        assert hasattr(x, 'shape'), 'Input x should be Array, not {}'.format(type(x))

        node_num = y.shape[-1]
        self.svr_list = [SVR(*args, **kwargs) for _ in range(node_num)]
        print('Start training...')
        for i in range(node_num):
            self.svr_list[i].fit(x[:, :, i], y[:, i])
            cur_pos = (i + 1) * 50 // node_num
            print('\r{}  {:.1%}'.format('▇' * cur_pos, (i + 1) / node_num), end='')
        print()

    def predict(self, x):
        assert hasattr(x, 'shape'), 'Input x should be Array, not {}'.format(type(x))
        assert self.svr_list is not None, 'The model shoud be fit first!'

        out = []
        for i in range(x.shape[-1]):
            out.append(self.svr_list[i].predict(x[:, :, i]))
        return np.array(out).T
