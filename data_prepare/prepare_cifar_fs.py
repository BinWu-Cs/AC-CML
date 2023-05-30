import json
import os
import shutil

from data_generate.split_generator import SplitGenerator

if __name__ == "__main__":
    # load config file
    # config_name = 'ptl_bomla_lam1.json'
    # jsonfile = open(os.path.join('./config/la_seqdataset', config_name))
    config_name = 'seqdataset.json'
    jsonfile = open(os.path.join('../config/vi_seqdataset', config_name))
    config = json.loads(jsonfile.read())

    # split_dir = os.path.join(os.path.join(config['data_dir'], config['split_folder']), 'cifar_fs')
    split_dir = os.path.join(os.path.join(config['data_dir'], 'cifar_fs', config['split_folder']))
    dest_dir = os.path.join(config['data_dir'], 'cifar_fs')

    # or use bertinetto's split
    # os.makedirs(split_dir)

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir)

    metatrain = [os.path.join(os.path.join(config['data_dir'], 'cifar100_raw', "data"), line.rstrip('\n'))
                 for line in open('../data/cifar100_raw/splits/bertinetto/train.txt', 'r')]
    metaval = [os.path.join(os.path.join(config['data_dir'], 'cifar100_raw', "data"), line.rstrip('\n'))
               for line in open('../data/cifar100_raw/splits/bertinetto/val.txt', 'r')]
    metatest = [os.path.join(os.path.join(config['data_dir'], 'cifar100_raw', "data"), line.rstrip('\n'))
                for line in open('../data/cifar100_raw/splits/bertinetto/test.txt', 'r')]

    import numpy as np
    np.save(os.path.join(split_dir, 'metatrain.npy'), metatrain)
    np.save(os.path.join(split_dir, 'metaval.npy'), metaval)
    np.save(os.path.join(split_dir, 'metatest.npy'), metatest)
