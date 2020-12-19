'''
based on places_diy, add a extra scene class and concate with inout dataset
'''
import sys
sys.path.append('./')
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from easydict import EasyDict as edict
import itertools
from torchvision import transforms as T
import random

PLACES_ROOT = '/home/luojiapeng/datasets/Places'

from dataloader.label_cfg_v2 import LABEL_CFG, INOUT_CFG
from dataloader.places_diy import PlacesDIY, read_labels_txt, get_cato_images, ConcatDataset

class PlacesDIY_V2(torch.utils.data.Dataset):
    def __init__(self, label_cfg, mode, img_tfm=None, label_tfm=None, root=PLACES_ROOT):
        assert mode in ['train', 'eval', 'test']
        self.label_cfg = label_cfg
        self.mode = mode
        self.img_tfm = img_tfm
        self.label_tfm = label_tfm
        self.images = []
        self.labels = []
        self.labelnames = list(label_cfg.keys())
        if mode == 'train':
            images_dir = Path(PLACES_ROOT) / 'data_256'
            for lname, v in label_cfg:
                for p in v:
                    if isinstance(p, str):
                        imgs = list((images_dir / p).glob('*.jpg'))
                    elif isinstance(p, Path):
                        imgs = list((p/'train').glob('*.jpg'))
                    else:
                        raise ValueError("wrong value in label_cfg")
                    self.images.extend(imgs)
                    self.labels.extend([self.labelnames.index(lname)] * len(imgs))
        elif mode == 'eval':
            self.images_dir = Path(PLACES_ROOT) / 'val_256'
            idx2name = self.create_idx2name()
            for line in  open(Path(PLACES_ROOT)/'places365_val.txt').readlines():
                keys = line.rstrip().split()
                if len(keys) != 2:
                    break
                name = idx2name[int(keys[1])]
                if name in self.labelnames:
                    self.images.append(self.images_dir / keys[0])
                    self.labels.append(self.labelnames.index(name))
            for lname, v in label_cfg:
                for p in v:
                    if isinstance(p, Path):
                        imgs = list((p/'val').glob('*.jpg'))
                    self.images.extend(imgs)
                    self.labels.extend([self.labelnames.index(lname)] * len(imgs))
        elif mode == 'test':
            raise NotImplementedError()
    
    def create_idx2name(self):
        cname2lname = {cname: lname for lname,cnames in self.label_cfg.items() for cname in cnames}
        idx2name = {}
        for line in open(Path(PLACES_ROOT) / 'categories_places365.txt').readlines():
            line = line.rstrip()
            if line:
                name, idx = line.split()
                idx2name[int(idx)] = cname2lname[name]
        return idx2name


    def __getitem__(self, k):
        image = Image.open(self.images[k]).convert('RGB')
        label = self.labels[k]
        if self.img_tfm is not None:
            image = self.img_tfm(image)
        if self.label_tfm is not None:
            label = self.label_tfm(label)
        return image, label
    
    def __len__(self):
        return len(self.images)

def label_tfm(x):
    return x + 2
def get_places_diy_loader(mode, size, batch_size, num_workers):
    MAX_SIZE = 320
    assert size < MAX_SIZE
    if mode == 'train':
        transform = T.Compose([T.Resize(MAX_SIZE), T.RandomCrop(size), T.ToTensor()])
        shuffle = True
    else:
        transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
        shuffle = False
    label_dataset = PlacesDIY(LABEL_CFG, mode, transform, label_tfm)
    inout_dataset = PlacesDIY(INOUT_CFG, mode, transform)
    dataset = ConcatDataset([label_dataset, inout_dataset], shuffle=shuffle)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader