import os, sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from setproctitle import *

from config import Config
from model.load_network import load_networks
from tqdm import tqdm


def setup(opt):
    # -------------------------------------------- setup device --------------------------------------------
    if len(opt.gpu_id) != 0:
        device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # -------------------------------------------- setup dataload --------------------------------------------
    if not opt.debugging:
        from ssun_server2.data.dataload import thinfilm_test
    else:
        from data.dataload import thinfilm_test

    dataload = thinfilm_test(opt.dataset_path)
    dataload_train = data.DataLoader(dataset=dataload, batch_size=opt.batch_size,
                                     shuffle=False,
                                     num_workers=opt.num_workers,
                                     drop_last=True, pin_memory=False)
    # -------------------------------------------- setup network --------------------------------------------
    from model.LNmodel import LN_4, LN_8, LN_12
    model_LN = LN_4(opt.network_type).to(device)
    # model_LN = LN_8(opt.network_type).to(device)
    # model_LN = LN_12(opt.network_type).to(device)

    if opt.continue_train:
         model_LN = load_networks(model_LN, opt.checkpoint_name, device, net_name='model_LN', weight_path="/storage/mskim/checkpoint/")

    else:
        NotImplementedError('Invalid network_type')
    # -------------------------------------------- setup optimizer --------------------------------------------
    if opt.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model_LN.parameters(), lr=opt.lr)
    elif opt.optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model_LN.parameters(), lr=opt.lr)
    else:
        optimizer = None
        NotImplementedError('{} not implemented'.format(opt.optimizer_name))
    # -------------------------------------------- setup scheduler --------------------------------------------
    if opt.scheduler_name == 'cycliclr':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=opt.lr, cycle_momentum=False, step_size_up=40,
                                          step_size_down=60, mode='exp_range', gamma=0.9)
    elif opt.scheduler_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-9)
    else:
        scheduler = None
        NotImplementedError('{} not implemented'.format(opt.scheduler_name))

    return device, dataload_train, model_LN, optimizer, scheduler


def test_reg(config, device, dataload_train, model_LN, optimizer, scheduler):
    for batch_id, data in enumerate(dataload_train, 1):
        with torch.no_grad():
            input = data['X'].float().to(device)
            label = data['Y'].float().to(device)




def test_class(config, device, dataload_train, model_LN, optimizer, scheduler):

    for batch_id, data in enumerate(dataload_train, 1):

        with torch.no_grad():
            input = data['X'].float().to(device)
            label = data['Y'].float().to(device)

            label_one_hot_1 = data['Y_one_hot_1'].type(torch.LongTensor).to(device)
            label_one_hot_2 = data['Y_one_hot_2'].type(torch.LongTensor).to(device)
            label_one_hot_3 = data['Y_one_hot_3'].type(torch.LongTensor).to(device)
            label_one_hot_4 = data['Y_one_hot_4'].type(torch.LongTensor).to(device)

            output_1, output_2, output_3, output_4 = model_LN.forward(input)

            one_hot = [label_one_hot_1, label_one_hot_2, label_one_hot_3, label_one_hot_4]
            outs= [output_1, output_2, output_3, output_4]

            pred = model_LN.predicted(outs)
            acc = model_LN.accuracy(pred, one_hot)

            print(acc)



if __name__ == '__main__':
    config = Config()
    config.print_options()

    setproctitle(config.opt.tensor_name)
    torch.cuda.set_device(int(config.opt.gpu_id))

    device, dataload_train, model_LN, optimizer, scheduler = setup(config.opt)

    if config.opt.network_type == 'reg':
        test_reg(config, device, dataload_train, model_LN, optimizer, scheduler)

    elif config.opt.network_type == 'class':
        test_class(config, device, dataload_train, model_LN, optimizer, scheduler)

    else:
        NotImplementedError()

