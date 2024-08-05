import os, sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ssun.config import Config
class Linear(nn.Module):
    def __init__(self, in_features=226, out_features=904):
        super(Linear, self).__init__()
        self.linear = nn.Sequential(nn.Linear(in_features, 452, bias=False),
                                    nn.BatchNorm1d(452),
                                    nn.ReLU(),
                                    nn.Linear(452, 452, bias=False),
                                    nn.BatchNorm1d(452),
                                    nn.ReLU(),
                                    nn.Linear(452, out_features, bias=False),
                                    nn.BatchNorm1d(out_features),
                                    nn.ReLU())

    def forward(self,x):
        return self.linear(x)

class Primarycaps(nn.Module):
    def __init__(self, in_features: int =904, out_features: int =4, cap_dims: int = 8):
        super(Primarycaps, self).__init__()
        assert in_features % out_features == 0

        self.out_features = int(out_features)
        self.cap_dims = cap_dims
        self.in_features = int(in_features)
        self.num_primary_caps = int(self.in_features / self.out_features)

        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=4, out_channels=self.cap_dims, kernel_size=[4, 1], stride=1, padding=0) for _ in range(self.num_primary_caps)])


    def forward(self, x):
        out = x.view(x.size(0), 4, -1).unsqueeze(dim=-1)
        out = [capsule(out) for capsule in self.capsules]
        out = torch.stack(out, dim=2)
        num_routes = out.shape[2]*out.shape[3]*out.shape[4]
        out = out.view(x.size(0), num_routes, -1)

        return self.squash(out)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1.+squared_norm) * torch.sqrt(squared_norm) + 1e-9)
        return output_tensor

class Digitcaps(nn.Module):
    def __init__(self, in_dim=4, in_caps=226, out_dim=8, out_caps=8, num_routing=3):
        super(Digitcaps, self).__init__()

        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing

        self.W = nn.Parameter(0.01 * torch.randn(in_caps, out_caps, in_dim, out_dim)) # torch.Size([50398, 4, 8, 30])

    def forward(self, x):
        batch_size = x.size(0)
        u_hat = torch.einsum('ijnm, bin -> bijm', self.W, x) # torch.Size([200, 50398, 4, 30])
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.in_caps, self.out_caps).to(x.device)
        for route_iter in range(self.num_routing - 1):
            c = F.softmax(b, dim=2)
            s = torch.einsum('bij,bijm->bjm', c, temp_u_hat)
            v = self.squash(s)
            a = torch.einsum('bjm,bijm->bij', v, temp_u_hat)
            b = b + a

        c = F.softmax(b, dim=2)
        s = torch.einsum('bij,bijm->bjm', c, u_hat)
        v = self.squash(s)
        return v

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm) + 1e-9)
        return output_tensor

class Decoder(nn.Module):
    def __init__(self, input_width=52, input_height=52, input_channel=3, out_caps=16, num_classes=8):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.out_caps = out_caps
        self.num_classes = num_classes
        self.reconstraction_layers = nn.Sequential(nn.Linear(self.out_caps * self.num_classes, 512),
                                                   nn.ReLU(),
                                                   nn.Linear(512, 1024),
                                                   nn.ReLU(),
                                                   nn.Linear(1024, self.input_width * self.input_height * self.input_channel),
                                                   nn.Sigmoid())

    def forward(self, x):
        classes = torch.sqrt((x**2).sum(dim=2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(8)).to(x.device)

        masked = masked.index_select(dim=0, index=Variable(max_length_indices.data))
        t = (x * masked[:, :, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked

class capsnet(nn.Module):
    def __init__(self, opt):
        super(capsnet, self).__init__()
        self.in_features = 226
        self.out_features = 904
        self.loss_name = nn.MSELoss()

        self.linear_layer = Linear(in_features=self.in_features, out_features=self.out_features)
        self.primary_layer = Primarycaps(in_features=self.out_features, out_features=(self.out_features/self.in_features))
        self.digit_capsules = Digitcaps(in_dim=8, in_caps=50398, out_dim=30, out_caps=4, num_routing=3)
        # self.decoder = Decoder(input_width=opt.data_height, input_height=opt.data_width, input_channel=opt.data_depth, out_caps=opt.out_dim, num_classes=8)


    def forward(self, input):
        self.output_linear = self.linear_layer(input) # torch.Size([200, 904])
        self.output_primary = self.primary_layer(self.output_linear) # torch.Size([200, 50398, 8])
        self.output_digit = self.digit_capsules(self.output_primary) # torch.Size([200, 4, 30])

        return self.output_digit

    def predicted(self, output):
        predict = []
        for i in range(4):
            predict.append(torch.softmax(output[:, i, :], dim=1))
        return predict

    def accuracy(self, pred, target):
        acc_layer = []
        for i in range(4):
            acc_layer.append(torch.sum(torch.argmax(pred[i], dim=1) == torch.argmax(target[i], dim=1)) / float(pred[0].shape[0]))
        acc = sum(acc_layer) / float(len(acc_layer))
        return float(acc.cpu().numpy())



if __name__ == '__main__':
    print('Debug StackedAE')
    config = Config()
    model = capsnet(config.opt)
    x = torch.rand(200,226)
    output = model.forward(x)