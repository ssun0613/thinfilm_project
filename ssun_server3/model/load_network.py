import torch
import torch.nn as nn

def load_networks(net, checkpoint_name, device, net_name, weight_path=None):
    load_filename = '{}.pth'.format(checkpoint_name)
    if weight_path is None:
        ValueError('Should set the weight_path, which is the path to the folder including weights')
    else:
        load_path = weight_path + load_filename
    net = net
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    print('loading the model from %s' % load_path)
    state_dict = torch.load(load_path, map_location=str(device))

    ##  Load weights per layer
    net_layers = net.state_dict().keys()
    for name in net_layers:
        if name in list(state_dict['model_LN'].keys()):
            if net.state_dict()[name].shape == state_dict['model_LN'][name].shape:
                print(f"{name} weight Loaded...")
                net.state_dict()[name].copy_(state_dict['model_LN'][name])

    # net.load_state_dict(state_dict['{}'.format(net_name)], strict=False)

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
        net.load_state_dic(state_dict['{}'.format(net_name)])
    print('load completed {}'.format(net_name))

    return net

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            import scipy.stats as stats
            stddev = m.stddev if hasattr(m, 'stddev') else 0.1
            X = stats.truncnorm(-2, 2, scale=stddev)
            values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
            values = values.view(m.weight.size())
            with torch.no_grad():
                m.weight.copy_(values)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

