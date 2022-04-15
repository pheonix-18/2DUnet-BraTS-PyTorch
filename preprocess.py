import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import nibabel as nib
import pickle
from tqdm import tqdm
import random
from torchvision.transforms import transforms
from scipy import ndimage
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_imgs = '/home/sarucrcv/datasets/Task01_BrainTumour/imagesTr/'
train_labels = '/home/sarucrcv/datasets/Task01_BrainTumour/labelsTr/'

    
    

modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
        'tpath': '/home/sarucrcv/projects/brats/train_pkl_all/', #Target Path of pickle files
        'root': '/home/sarucrcv/datasets/Task01_BrainTumour/', 
        'flist': '../3dunet/train.txt',
        'has_label': True
        }

# test/validation data
valid_set = {
        'tpath': '/home/sarucrcv/projects/brats/val_pkl_all/',
        'root': '/home/sarucrcv/datasets/Task01_BrainTumour/',
        'flist': '../3dunet/val.txt',
        'has_label': True
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data



def process_f32b0(tpath, img_path, label_path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    
    label = np.array(nib_load(label_path), dtype='uint8', order='C')
    images = np.stack(np.array(nib_load(img_path), dtype='float32', order='C'))  # [240,240,155]
    foldername = img_path.split('/')[-1]
    mask = images.sum(-1) > 0

    for k in range(155):
        foldername = foldername.split('.')[0]
        filename = foldername + "_"+str(k)+".npy"
        import pdb
        output = os.path.join(tpath, foldername) 
        os.makedirs(output, exist_ok=True)
        output = os.path.join(tpath, foldername , filename)
        with open(output, 'wb') as f:

            # pdb.set_trace()
            if has_label:
                pickle.dump((images[:,:,k,:], label[:,:,k]), f)
            else:
                pickle.dump(images[:,:,k,:], f)
            f.close()

        if not has_label:
            return


def doit(dset):
    root, has_label = dset['root'], dset['has_label']
    file_list = os.path.join(dset['flist'])
    names = open(file_list).read().splitlines()
    imgs_paths = [os.path.join(root, 'imagesTr', name) for name in names]
    labels_paths = [os.path.join(root, 'labelsTr', name) for name in names]

    for i in tqdm(range(len(imgs_paths))):
        process_f32b0(dset['tpath'],imgs_paths[i], labels_paths[i], has_label)

doit(train_set)
doit(valid_set)
