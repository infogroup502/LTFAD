import argparse
from torch.backends import cudnn
from utils.utils import *
from solver import Solver
import time
import warnings
warnings.filterwarnings('ignore')
import sys

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# setting seed
set_seed(42)

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(array[idx-1])


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
        solver.test()
    if config.mode == 'test':
        solver.test()

    return solver

if __name__ == '__main__':

    def list_type(arg):
        return [int(item) for item in arg.split(',')]

    parser = argparse.ArgumentParser()

    # Alternative
    parser.add_argument('--win_size', type=int, default=90)
    parser.add_argument('--local_size', type=list_type, default=[5])
    parser.add_argument('--global_size', type=list_type, default=[15])
    parser.add_argument('--anormly_ratio', type=float, default=0.98)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--r', type=float, default=0.5)
    parser.add_argument('--d_model', type=int, default=128)


    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
    parser.add_argument('--loss_fuc', type=str, default='MSE')
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--input_c', type=int, default=1)
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')



    config = parser.parse_args()
    args = vars(config)
    config.local_size = [int(index) for index in config.local_size]

    
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ','')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]
    
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)

    sys.stdout = Logger("result/"+ config.data_path +".log", sys.stdout)
    if config.mode == 'train':
        print("\n\n")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('================ Hyperparameters ===============')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('====================  Train  ===================')
        
    main(config)

    
