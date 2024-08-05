import torch
import torch.nn as nn
import torch.nn.functional as F

class LN_4(nn.Module):
    def __init__(self, network_type):
        super(LN_4, self).__init__()
        self.network_type = network_type
        self.last_layer = 240
        self.linear_model = nn.Sequential(nn.Linear(226, 452, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(452, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, self.last_layer, bias=False),
                                          nn.ReLU())

        self.reg_layer = nn.Linear(self.last_layer, 4, bias=False)
        self.layer_1 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_2 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_3 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_4 = nn.Linear(self.last_layer, 30, bias=False)

    def forward(self, input):
        if self.network_type == 'reg':
            output = self.linear_model(input)
            output = self.reg_layer(output)
            return output

        elif self.network_type == 'class':
            output = self.linear_model(input)
            output_1 = self.layer_1(output)
            output_2 = self.layer_2(output)
            output_3 = self.layer_3(output)
            output_4 = self.layer_4(output)
            return output_1, output_2, output_3, output_4

    def predicted(self, output):
        predict = []
        for i in range(4):
            predict.append(torch.softmax(output[i], dim=1))
        return predict

    def accuracy(self, pred, target):
        acc_layer = []

        for i in range(4):
            acc_layer.append(torch.sum(torch.argmax(pred[i], dim=1) == torch.argmax(target[i], dim=1)) / float(pred[0].shape[0]))

        acc = sum(acc_layer) / float(len(acc_layer))

        return float(acc.cpu().numpy())

class LN_8(nn.Module):
    def __init__(self, network_type):
        super(LN_8, self).__init__()
        self.network_type = network_type
        self.last_layer = 240
        self.linear_model = nn.Sequential(nn.Linear(226, 452, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(452, 452, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(452, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, self.last_layer, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(self.last_layer, self.last_layer, bias=False),
                                          nn.ReLU())

        self.reg_layer = nn.Linear(self.last_layer, 4, bias=False)
        self.layer_1 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_2 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_3 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_4 = nn.Linear(self.last_layer, 30, bias=False)

    def forward(self, input):
        if self.network_type == 'reg':
            output = self.linear_model(input)
            output = self.reg_layer(output)
            return output

        elif self.network_type == 'class':
            output = self.linear_model(input)
            output_1 = self.layer_1(output)
            output_2 = self.layer_2(output)
            output_3 = self.layer_3(output)
            output_4 = self.layer_4(output)
            return output_1, output_2, output_3, output_4

    def predicted(self, output):
        predict = []
        for i in range(4):
            predict.append(torch.softmax(output[i], dim=1))
        return predict

    def accuracy(self, pred, target):
        acc_layer = []

        for i in range(4):
            acc_layer.append(torch.sum(torch.argmax(pred[i], dim=1) == torch.argmax(target[i], dim=1)) / float(pred[0].shape[0]))

        acc = sum(acc_layer) / float(len(acc_layer))

        return float(acc.cpu().numpy())
class LN_12(nn.Module):
    def __init__(self, network_type):
        super(LN_12, self).__init__()
        self.network_type = network_type
        self.last_layer = 240
        self.linear_model = nn.Sequential(nn.Linear(226, 452, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(452, 452, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(452, 452, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(452, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, 226, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(226, self.last_layer, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(self.last_layer, self.last_layer, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(self.last_layer, self.last_layer, bias=False),
                                          nn.ReLU())

        self.reg_layer = nn.Linear(self.last_layer, 4, bias=False)
        self.layer_1 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_2 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_3 = nn.Linear(self.last_layer, 30, bias=False)
        self.layer_4 = nn.Linear(self.last_layer, 30, bias=False)

    def forward(self, input):
        if self.network_type == 'reg':
            output = self.linear_model(input)
            output = self.reg_layer(output)
            return output

        elif self.network_type == 'class':
            output = self.linear_model(input)
            output_1 = self.layer_1(output)
            output_2 = self.layer_2(output)
            output_3 = self.layer_3(output)
            output_4 = self.layer_4(output)
            return output_1, output_2, output_3, output_4

    def predicted(self, output):
        predict = []
        for i in range(4):
            predict.append(torch.softmax(output[i], dim=1))
        return predict

    def accuracy(self, pred, target):
        acc_layer = []

        for i in range(4):
            acc_layer.append(torch.sum(torch.argmax(pred[i], dim=1) == torch.argmax(target[i], dim=1)) / float(pred[0].shape[0]))

        acc = sum(acc_layer) / float(len(acc_layer))

        return float(acc.cpu().numpy())
