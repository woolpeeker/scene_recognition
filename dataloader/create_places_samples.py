import numpy as np
from pathlib import Path
from PIL import Image
import torch
from easydict import EasyDict as edict
import random
import shutil

OUT_ROOT = Path('/home/luojiapeng/datasets/Places/samples')

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

if __name__ == '__main__':
    std_places_root = Path('/home/luojiapeng/datasets/Places')
    std_places_txt = std_places_root /  'categories_places365.txt'
    std_places_labels = read_labels_txt(std_places_txt)
    images_dir = std_places_root / 'data_256'
    
    label_names = read_labels_txt(std_places_txt)
    images, labels = get_data_lst(images_dir, label_names)

    label_idxs = [[] for _ in range(len(label_names))]
    for i, label in enumerate(labels):
        label_idxs[label].append(i)
    print('>>> start copy images')
    for label, idxs in enumerate(label_idxs):
        chosen_idxs = random.choices(idxs, k=50)
        for idx in chosen_idxs:
            img_file = images[idx]
            out_img_file = Path(str(img_file).replace(str(images_dir), str(OUT_ROOT)))
            out_img_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(str(img_file), str(out_img_file))
        
        print(f">>> {label_names[label]} finished")
