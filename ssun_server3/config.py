import argparse
import os

def dataset_info(network_name):
    dataset_info = dict()
    dataset_info['dataset_path'] = '/storage/mskim/thinfilm/'
    dataset_info['batch_size'] = 100

    return dataset_info

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--network_name', type=str, default='thinfilm_LN')
        self.parser.add_argument('--network_type', type=str, default='class', help='[class | reg]')
        self.parser.add_argument('--weight_name', type=str, default='thinfilm_LN')
        self.parser.add_argument('--dataset_name', type=str, default='thinfilm')
        self.parser.add_argument('--tensor_name', type=str, default='thinfilm_0')
        self.parser.add_argument('--checkpoint_name', type=str, default='thinfilm_0')
        self.parser.add_argument('--continue_train', type=bool, default=False)
        self.parser.add_argument('--epochs', type=int, default=1750)
        self.parser.add_argument('--checkpoint_load_num', type=int, default=500)

        parser, _ = self.parser.parse_known_args()
        self.dataset_info = dataset_info(network_name=parser.network_name)

        self.parser.add_argument('--batch_size', type=int, default=self.dataset_info['batch_size'])
        self.parser.add_argument('--dataset_path', type=str, default=self.dataset_info['dataset_path'])

        self.parser.add_argument('--scheduler_name', type=str, default='cosine', help='[stepLR | cycliclr | cosine]')
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--optimizer_name', type=str, default='Adam', help='[Adam | RMSprop]')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='monument for rmsprop optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

        self.parser.add_argument('--save_path', type=str, default='./checkpoints/pre_test_{}'.format(parser.network_name), help='path to store model')
        self.parser.add_argument('--train_test_save_path', type=str, default='./train_test/' + parser.network_name, help='')
        self.parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
        self.parser.add_argument('--gpu_id', type=str, default='0', help='gpu id used to train')
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--num_workers', type=int, default=10)
        self.parser.add_argument('--samplier', type=int, default=1)
        self.parser.add_argument('--debugging', type=bool, default=False)
        self.parser.add_argument('--num_test_iter', type=int, default=5)

        self.opt, _ = self.parser.parse_known_args()

    def print_options(self):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(self.opt.save_path)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(self.opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
