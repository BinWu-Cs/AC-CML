import os
import json
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import glob
import shutil
import cv2

def save_class_images(root_dir, label, images, images_name):

    lbl_dir = os.path.join(root_dir, label)
    if not os.path.exists(lbl_dir):
        os.mkdir(lbl_dir)
    
    for image, name in zip(images, images_name):
        im = Image.fromarray(image)
        im.save(os.path.join(lbl_dir, name))


def augment_cls(root_dir, label):

    class_dir = os.path.join(root_dir, label) 

    for r in [90, 180, 270]:

        imgdir_rotate = os.path.join(root_dir, "{}_rotate{}".format(label, r))
        os.mkdir(imgdir_rotate)

        for imgname in os.listdir(class_dir):

            img = cv2.imread(os.path.join(class_dir, imgname))
            (height, width) = img.shape[:-1]
            center = (height/2, width/2)

            rotate = cv2.getRotationMatrix2D(center, r, 1.0)
            img_rotate = cv2.warpAffine(img, rotate, (height, width))
            cv2.imwrite(os.path.join(imgdir_rotate, 'rot{}_'.format(r) + imgname), img_rotate)


if __name__ == '__main__':
    # load config file
    """
    config_name = 'oqc_bomla_lam1.json'
    jsonfile = open(os.path.join('../config/la_seqdataset', config_name))
    config = json.loads(jsonfile.read())

    dest_dir = os.path.join(config['data_dir'], 'mini_imagenet')
    # split_dir = os.path.join(os.path.join(config['data_dir'], config['split_folder']), 'mini_imagenet')
    split_dir = os.path.join(os.path.join(config['data_dir'], 'mini_imagenet', config['split_folder']))
    """
    data_dir = "../data"
    split_folder = "split"
    dest_dir = os.path.join(data_dir, 'mini_imagenet')
    split_dir = os.path.join(os.path.join(data_dir, 'mini_imagenet', split_folder))

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir)

    # use split in Ravi & Larochelle
    #metatrain = [os.path.join(dest_dir, clsname)
    #             for clsname in pd.read_csv('../data/mini_imagenet_raw/train.csv')['label'].unique()]
    #metaval = [os.path.join(dest_dir, clsname)
    #           for clsname in pd.read_csv('../data/mini_imagenet_raw/val.csv')['label'].unique()]
    #metatest = [os.path.join(dest_dir, clsname)
    #            for clsname in pd.read_csv('../data/mini_imagenet_raw/test.csv')['label'].unique()]
    train = pd.read_csv('../data/mini_imagenet_raw/train.csv')
    val = pd.read_csv('../data/mini_imagenet_raw/val.csv')
    test = pd.read_csv('../data/mini_imagenet_raw/test.csv')

    metatrain_lbl = [clsname for clsname in train['label'].unique()]
    metaval_lbl = [clsname for clsname in val['label'].unique()]
    metatest_lbl = [clsname for clsname in test['label'].unique()]

    # generate split folder
    train_dir = os.path.join(dest_dir, "metatrain")
    val_dir = os.path.join(dest_dir, "metaval")
    test_dir = os.path.join(dest_dir, "metatest")

    # remove all classes if train, val, test non empty
    if os.path.exists(train_dir):
        print('Train destination folder not empty. Deleting...')
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        print('Val destination folder not empty. Deleting...')
        shutil.rmtree(val_dir)
    if os.path.exists(test_dir):
        print('Test destination folder not empty. Deleting...')
        shutil.rmtree(test_dir)

    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    # save meta train images
    train_data = pickle.load(open("../data/mini_imagenet_raw/mini-imagenet-cache-train.pkl", mode="rb"))
    val_data = pickle.load(open("../data/mini_imagenet_raw/mini-imagenet-cache-val.pkl", mode="rb"))
    test_data = pickle.load(open("../data/mini_imagenet_raw/mini-imagenet-cache-test.pkl", mode="rb"))
    train_images = train_data["image_data"] # N, h, w, c
    train_class_dict = train_data["class_dict"]
    val_images, val_class_dict = val_data["image_data"], val_data["class_dict"]
    test_images, test_class_dict = test_data["image_data"], test_data["class_dict"]

    for lbl in metatrain_lbl:
        save_class_images(train_dir, lbl, train_images[train_class_dict[lbl]], 
                            train[train['label'] == lbl]["filename"])
    print("successfully save metatrain images.....")

    for lbl in metaval_lbl:
        save_class_images(val_dir, lbl, val_images[val_class_dict[lbl]], 
                            val[val['label'] == lbl]["filename"])
    print("successfully save metaval images.....")
    
    for lbl in metatest_lbl:
        save_class_images(test_dir, lbl, test_images[test_class_dict[lbl]], 
                            test[test['label'] == lbl]["filename"])
    print("successfully save metatest images.....")

    # agument the class
    """
    for train_lbl in os.listdir(train_dir):
        augment_cls(train_dir, train_lbl)
    print("successfully agument metatrain images.....")
    
    for val_lbl in os.listdir(val_dir):
        augment_cls(val_dir, val_lbl)
    print("successfully agument metaval images.....")
    
    for test_lbl in os.listdir(test_dir):
        augment_cls(test_dir, test_lbl)
    print("successfully agument metatest images.....")
    """

    np.save(os.path.join(split_dir, 'metatrain.npy'), glob.glob(train_dir + '/*'))
    np.save(os.path.join(split_dir, 'metaval.npy'), glob.glob(val_dir + '/*'))
    np.save(os.path.join(split_dir, 'metatest.npy'), glob.glob(test_dir + '/*'))
