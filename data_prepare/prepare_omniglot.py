import os
import json

from data_generate.split_generator import SplitGenerator

if __name__ == "__main__":
    # load config file
    config_name = 'seqdataset.json'
    jsonfile = open(os.path.join('../config/vi_seqdataset', config_name))
    config = json.loads(jsonfile.read())

    #split_dir = os.path.join(os.path.join(config['data_dir'], config['split_folder']), 'omniglot')
    split_dir = os.path.join(os.path.join(config['data_dir'], 'omniglot', config['split_folder']))
    dest_dir = os.path.join(config['data_dir'], 'omniglot')

    split_omniglot = SplitGenerator(
        data_dir=os.path.join(config['data_dir'], 'omniglot_raw'), dest_dir=dest_dir, split_dir=split_dir,
        back_eval_raw=True, supercls_raw=True, supercls_split=config['omniglot']['supercls']
    )
    split_omniglot.split_train_val_test(nclass_train=1100, nclass_val=100, save_split_npy=False, csv_save_form=None)
    split_omniglot.generate_foldersplit(save_split_npy=True)
    # split_omniglot.augment_cls(type='rotation', save_split_npy=True)
