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

PLACES_ROOT = '/data/luojiapeng/datasets/Places'

def read_labels_txt(txt_path):
    labels = []
    for line in open(txt_path).readlines():
        line = line.rstrip()
        if line:
            name, idx = line.split()
            idx = int(idx)
            assert idx == len(labels)
            assert name[0]
            labels.append(name)
    return labels

def get_cato_images(images_root, cato_name):
    p = Path(images_root) / ('.'+ cato_name)
    return list(p.glob('*.jpg'))


class PlacesDIY(torch.utils.data.Dataset):
    def __init__(self, label_cfg, mode, img_tfm=None, label_tfm=None):
        assert mode in ['train', 'eval', 'test']
        self.label_cfg = label_cfg
        self.mode = mode
        self.img_tfm = img_tfm
        self.label_tfm = label_tfm
        self.labels_txt = Path(PLACES_ROOT) / 'categories_places365.txt'
        self.places_cato_names = read_labels_txt(self.labels_txt)
        self.label_names = list(label_cfg.keys()) # large classes
        self.cname2lname = {cname: lname for lname,cnames in label_cfg.items() for cname in cnames}
        if mode == 'train':
            images_dir = Path(PLACES_ROOT) / 'data_256'
            images = [get_cato_images(images_dir, name) for name in self.cname2lname.keys()]
            labels = [[name] * len(images[i]) for i, name in enumerate(self.cname2lname.values())]
            labels = list(itertools.chain.from_iterable(labels))
            labels = [self.label_names.index(x) for x in labels]
            self.images = list(itertools.chain.from_iterable(images))
            self.labels = labels
        elif mode == 'eval':
            self.images_dir = Path(PLACES_ROOT) / 'val_256'
            images, labels = [], []
            for line in  open(Path(PLACES_ROOT)/'places365_val.txt').readlines():
                keys = line.rstrip().split()
                if len(keys) != 2:
                    break
                cato_idx = int(keys[1])
                cato_name = self.places_cato_names[cato_idx]
                if cato_name in self.cname2lname:
                    images.append(self.images_dir / keys[0])
                    labels.append(self.cname2lname[cato_name])
            labels = [self.label_names.index(x) for x in labels]
            self.images = images
            self.labels = labels
        elif mode == 'test':
            raise NotImplementedError()

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

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, shuffle):
        self.datasets = datasets
        self.idx_lst = list(range(len(self)))
        if shuffle:
            random.shuffle(self.idx_lst)

    def __getitem__(self, i):
        i = self.idx_lst[i]
        si = 0
        for d in self.datasets:
            if si + len(d) > i:
                break
            else:
                si += len(d)
        return d[i - si]

    def __len__(self):
        return sum(len(d) for d in self.datasets)


from dataloader.label_cfg import LABEL_CFG, INOUT_CFG

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

if __name__ == '__main__':
    loader = get_places_diy_loader('eval', 224, 32, 0)
    for i, data in enumerate(loader):
        pass
