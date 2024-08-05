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
from model.load_network import load_networks, count_parameters
from tqdm import tqdm


def setup(opt):
    # -------------------------------------------- setup device --------------------------------------------
    if len(opt.gpu_id) != 0:
        device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    # -------------------------------------------- setup dataload --------------------------------------------
    if not opt.debugging:
        from ssun.data.dataload import thinfilm
    else:
        from data.dataload import thinfilm

    dataload = thinfilm(opt.dataset_path)
    dataload_train = data.DataLoader(dataset=dataload, batch_size=opt.batch_size,
                                     shuffle=False,
                                     num_workers=opt.num_workers,
                                     drop_last=True, pin_memory=False)
    # -------------------------------------------- setup network --------------------------------------------
    from model.LNmodel import LN_4, LN_8, LN_12
    # model_LN = LN_4(opt.network_type).to(device)
    # model_LN = LN_8(opt.network_type).to(device)
    model_LN = LN_12(opt.network_type).to(device)


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

def train_reg(config, writer, device, dataload_train, model_LN, optimizer, scheduler):
    os.makedirs("/storage/mskim/checkpoint/{}".format(config.opt.checkpoint_name), exist_ok=True)

    print("\n-------------------------------------- train_reg --------------------------------------\n")
    loss_fn = nn.L1Loss()

    global_step = 0
    t_loss = []
    for curr_epoch in range(config.opt.epochs):
        print("--------------------------------------[ Epoch : {} ]-------------------------------------".format(curr_epoch + 1))
        loss = 0.0
        pbar = tqdm(enumerate(dataload_train, 1), leave=True, total=len(dataload_train), desc='Loss: {:.4f}'.format(loss))
        for batch_id, data in pbar:
            global_step += 1

            input = data['X'].float().to(device)
            label = data['Y'].float().to(device)

            output = model_LN.forward(input)

            loss = loss_fn(label, output)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            t_loss.append(loss.detach().cpu().numpy().item())
            pbar.set_description(desc='loss: {:.4f}'.format(loss.detach().cpu().numpy().item()))

        # if batch_id % 000 == 0:
        writer.add_scalar("loss", np.mean(t_loss), global_step)
        t_loss = []

        print("loss : %.5lf\n" % loss)
        scheduler.step()
        writer.close()

        if (curr_epoch + 1 == 250) or (curr_epoch + 1 == 750) or (curr_epoch + 1 == 1000):
            torch.save({'model_LN': model_LN.state_dict(), 'optimizer': optimizer.state_dict()}, "/storage/mskim/checkpoint/{}/{}_{}.pth".format(config.opt.checkpoint_name,config.opt.checkpoint_name, curr_epoch+1))


def train_class(config, writer, device, dataload_train, model_LN, optimizer, scheduler):
    os.makedirs("/storage/mskim/checkpoint/{}".format(config.opt.checkpoint_name), exist_ok=True)

    print("\n------------------------------------- train_class -------------------------------------\n")
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    t_loss = []
    t_acc = []
    for curr_epoch in range(config.opt.epochs):
        print("--------------------------------------[ Epoch : {} ]-------------------------------------".format(curr_epoch + 1))
        loss = 0.0
        pbar = tqdm(enumerate(dataload_train, 1), leave=True, total=len(dataload_train), desc='Loss: {:.4f}'.format(loss))
        for batch_id, data in pbar:
            global_step += 1

            input = data['X'].float().to(device)

            label_one_hot_1 = data['Y_one_hot_1'].type(torch.LongTensor).to(device)
            label_one_hot_2 = data['Y_one_hot_2'].type(torch.LongTensor).to(device)
            label_one_hot_3 = data['Y_one_hot_3'].type(torch.LongTensor).to(device)
            label_one_hot_4 = data['Y_one_hot_4'].type(torch.LongTensor).to(device)

            output_1, output_2, output_3, output_4 = model_LN.forward(input)

            one_hot = [label_one_hot_1, label_one_hot_2, label_one_hot_3, label_one_hot_4]
            outs= [output_1, output_2, output_3, output_4]

            loss = loss_fn(outs[0], one_hot[0].type(torch.FloatTensor).to(device))
            for layer_idx in range(1, 4):
                loss += loss_fn(outs[layer_idx], one_hot[layer_idx].type(torch.FloatTensor).to(device))
            #

            pred = model_LN.predicted(outs)
            acc = model_LN.accuracy(pred, one_hot)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            t_loss.append(loss.detach().cpu().numpy().item())
            t_acc.append(acc)

            pbar.set_description(desc='loss: {:.4f}'.format(loss.detach().cpu().numpy().item()))

        print(count_parameters(model_LN))

        if curr_epoch % 50 == 0:
            writer.add_scalar("loss", np.mean(t_loss), global_step)
            writer.add_scalar("acc", np.mean(t_acc), global_step)
            t_loss = []
            t_acc = []

        print("loss : %.5lf\n" % loss)
        scheduler.step()
        writer.close()

        if (curr_epoch+1 == 250) or (curr_epoch+1 == 750) or (curr_epoch+1 == 1250) or (curr_epoch+1 == 1750):
            torch.save({'model_LN': model_LN.state_dict(), 'optimizer': optimizer.state_dict()}, "/storage/mskim/checkpoint/{}/{}_{}.pth".format(config.opt.checkpoint_name, config.opt.checkpoint_name, curr_epoch+1))



if __name__ == '__main__':
    config = Config()
    config.print_options()

    setproctitle(config.opt.tensor_name)
    torch.cuda.set_device(int(config.opt.gpu_id))

    writer = SummaryWriter('/storage/mskim/tensorboard/{}'.format(config.opt.tensor_name))
    device, dataload_train, model_LN, optimizer, scheduler = setup(config.opt)

    if config.opt.network_type == 'reg':
        train_reg(config, writer, device, dataload_train, model_LN, optimizer, scheduler)

    elif config.opt.network_type == 'class':
        train_class(config, writer, device, dataload_train, model_LN, optimizer, scheduler)

    else:
        NotImplementedError()

