import os
import numpy as np
import pandas as pd
import torch


class thinfilm():
    def __init__(self, path):
        super(thinfilm, self).__init__()
        train = pd.read_csv(path + 'train_ssun.csv').iloc[:, 1:]
        self.train_X, self.train_Y = train.iloc[:, 4:], train.iloc[:, 0:4]
        self.layer_1, self.layer_2, self.layer_3, self.layer_4 = train.iloc[:, 0:1].values, train.iloc[:, 1:2].values, train.iloc[:, 2:3].values, train.iloc[:, 3:4].values
        self.tmp_x, self.tmp_y = self.train_X.values, self.train_Y.values

    def __len__(self):
        return len(self.train_X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.tmp_x)[idx]
        Y = torch.from_numpy(self.tmp_y)[idx]

        layer_1_Y = torch.from_numpy(self.layer_1)[idx]
        layer_2_Y = torch.from_numpy(self.layer_2)[idx]
        layer_3_Y = torch.from_numpy(self.layer_3)[idx]
        layer_4_Y = torch.from_numpy(self.layer_4)[idx]

        one_hot_1 = self.one_hot_encoding(layer_1_Y)
        one_hot_2 = self.one_hot_encoding(layer_2_Y)
        one_hot_3 = self.one_hot_encoding(layer_3_Y)
        one_hot_4 = self.one_hot_encoding(layer_4_Y)

        return {'X': X, 'Y': Y, 'Y_one_hot_1': one_hot_1, 'Y_one_hot_2': one_hot_2, 'Y_one_hot_3': one_hot_3, 'Y_one_hot_4': one_hot_4}

    def one_hot_encoding(self, Y):

        one_hot = np.zeros(30)
        one_hot[int(Y / 10 - 1)] = 1.0

        return one_hot

class thinfilm_test():
    def __init__(self, path):
        super(thinfilm_test, self).__init__()
        train = pd.read_csv(path + 'test_ssun.csv').iloc[:, 1:]
        self.train_X, self.train_Y = train.iloc[:, 4:], train.iloc[:, 0:4]
        self.layer_1, self.layer_2, self.layer_3, self.layer_4 = train.iloc[:, 0:1].values, train.iloc[:, 1:2].values, train.iloc[:, 2:3].values, train.iloc[:, 3:4].values
        self.tmp_x, self.tmp_y = self.train_X.values, self.train_Y.values

    def __len__(self):
        return len(self.train_X)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.tmp_x)[idx]
        Y = torch.from_numpy(self.tmp_y)[idx]

        layer_1_Y = torch.from_numpy(self.layer_1)[idx]
        layer_2_Y = torch.from_numpy(self.layer_2)[idx]
        layer_3_Y = torch.from_numpy(self.layer_3)[idx]
        layer_4_Y = torch.from_numpy(self.layer_4)[idx]

        one_hot_1 = self.one_hot_encoding(layer_1_Y)
        one_hot_2 = self.one_hot_encoding(layer_2_Y)
        one_hot_3 = self.one_hot_encoding(layer_3_Y)
        one_hot_4 = self.one_hot_encoding(layer_4_Y)

        return {'X': X, 'Y': Y, 'Y_one_hot_1': one_hot_1, 'Y_one_hot_2': one_hot_2, 'Y_one_hot_3': one_hot_3, 'Y_one_hot_4': one_hot_4}

    def one_hot_encoding(self, Y):

        one_hot = np.zeros(30)
        one_hot[int(Y / 10 - 1)] = 1.0

        return one_hot

if __name__ == '__main__':
    data = thinfilm('/storage/mskim/thinfilm/csv/').__getitem__(0)
