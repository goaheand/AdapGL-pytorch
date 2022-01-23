import numpy as np
from torch.utils.data import Dataset


class TPDataset(Dataset):
    def __init__(self, data_path, keys=('x', 'y')):
        self.data_path = data_path
        self.keys = keys

        data_npz = np.load(data_path)
        self.data = {k: data_npz[k] for k in self.keys}

    def fit(self, scaler):
        key = self.keys[0]
        self.data[key] = scaler.transform(self.data[key])

        key = self.keys[-1]
        self.data[key] = scaler.transform(self.data[key], axis=0)

    def __getitem__(self, index: int):
        ret = [self.data[key][index] for key in self.keys]
        return ret

    def __len__(self):
        key = self.keys[-1]
        return self.data[key].shape[0]
