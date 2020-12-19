
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms as T

PLACES_ROOT = '/home/luojiapeng/datasets/Places'

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

def get_data_lst(root, label_names):
    # return image list and label list
    root = Path(root)
    images = []
    labels = []

    for i, name in enumerate(label_names):
        p = root / name[1:]
        for f in p.iterdir():
            images.append(f)
            labels.append(i)
    return images, labels

class PlacesStandard_256(torch.utils.data.Dataset):
    def __init__(self, mode, transform=None, labels=None):
        assert mode in ['train', 'eval', 'test']
        self.mode = mode
        self.transform = transform
        self.labels_txt = Path(PLACES_ROOT) / 'categories_places365.txt'
        self.all_label_names = read_labels_txt(self.labels_txt)
        if labels is None:
            self.label_names = self.all_label_names
        else:
            self.label_names = labels
        if mode == 'train':
            self.images_dir = Path(PLACES_ROOT) / 'data_256'
            images, labels = get_data_lst(self.images_dir, self.label_names)
            self.images = images
            self.labels = labels
        elif mode == 'eval':
            self.images_dir = Path(PLACES_ROOT) / 'val_256'
            images, labels = [], []
            for line in  open(Path(PLACES_ROOT)/'places365_val.txt').readlines():
                keys = line.rstrip().split()
                if len(keys) != 2:
                    break
                label_idx = int(keys[1])
                label_name = self.all_label_names[label_idx]
                if label_name in self.label_names:
                    label_idx = self.label_names.index(label_name)
                    labels.append(label_idx)
                    images.append(self.images_dir / keys[0])
            self.images = images
            self.labels = labels
        elif mode == 'test':
            raise NotImplementedError()

    def __getitem__(self, k):
        image = Image.open(self.images[k]).convert('RGB')
        label = self.labels[k]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.images)

def get_places_standard_256_loader(size, batch_size, num_workers, mode):
    MAX_SIZE = 320
    assert size < MAX_SIZE
    if mode == 'train':
        transform = T.Compose([T.Resize(MAX_SIZE), T.RandomCrop(size), T.ToTensor()])
    else:
        transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
    dataset = PlacesStandard_256(mode, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return loader

def get_places_10_loader(size, batch_size, num_workers, mode):
    labels = [
        '/a/airfield', '/a/art_school', '/b/bridge', '/c/crosswalk', '/d/downtown',
        '/a/army_base', '/f/fire_station', '/h/highway', '/g/gas_station', '/c/church/outdoor'
    ]
    MAX_SIZE = 320
    assert size < MAX_SIZE
    if mode == 'train':
        transform = T.Compose([T.Resize(MAX_SIZE), T.RandomCrop(size), T.ToTensor()])
        shuffle = True
    else:
        transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])
        shuffle = False
    dataset = PlacesStandard_256(mode, transform, labels=labels)
    print('get_places_10_loader length: %d' % len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader