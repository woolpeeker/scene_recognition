
import numpy as np
import torch
from PIL import Image

def save_fmap(fmap, path):
    if isinstance(fmap, torch.Tensor):
        fmap = fmap.detach().cpu().numpy()
    _min = np.min(fmap)
    _max = np.max(fmap)
    if _min == _max: _max+=1
    fmap = (fmap-_min) / (_max-_min) * 255
    Image.fromarray(fmap.astype(np.uint8)).save(path)
