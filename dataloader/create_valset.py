import sys
sys.path.append('./')
import numpy as np
from pathlib import Path
import shutil
from dataloader.label_cfg import LABEL_CFG, INOUT_CFG
from places_diy import read_labels_txt

PLACES_ROOT = Path('/home/luojiapeng/datasets/Places/')
IMG_DIR = PLACES_ROOT / 'val_256'
TXT_FILE = PLACES_ROOT / 'places365_val.txt'
LABEL_TXT = PLACES_ROOT / 'categories_places365.txt'

def func(cfg, out_dir:Path):
    data = np.loadtxt(TXT_FILE, dtype=str).tolist()
    labelnames = read_labels_txt(LABEL_TXT)
    cname2lname = {cname: lname for lname,cnames in cfg.items() for cname in cnames}
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = open(out_dir / 'labels.txt', 'w')
    for img_file, label in data:
        img_file = Path(IMG_DIR) / img_file
        name = labelnames[int(label)]
        if not name in cname2lname.keys():
            continue
        labelname = cname2lname[name]
        fp.write(f'{img_file.name} {labelname}\n')
        dest = out_dir / img_file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(img_file), str(dest))
    fp.close()


if __name__ == '__main__':
    func(LABEL_CFG, Path('data/valset/scenes'))
    func(INOUT_CFG, Path('data/valset/inoutdoor'))