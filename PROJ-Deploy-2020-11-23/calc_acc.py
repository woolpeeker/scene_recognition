import numpy as np
from pathlib import Path

PRED_TXT='preds.txt'
SCENE_TXT='/mnt/HD_2/luojiapeng/scene_recognition/data/valset/scenes/labels.txt'

if __name__ == '__main__':
    preds = np.loadtxt(PRED_TXT,dtype=str).tolist()
    correct = 0
    total = 0
    gt = np.loadtxt(SCENE_TXT, dtype=str).tolist()
    gt = dict(gt)

    for p in preds:
        img_name = Path(p.split(':')[0]).name
        pred_scene = p.split(':')[1].split(',')[1]
        if pred_scene == gt[img_name]:
            correct += 1
        total += 1
    print('scene accuracy: %.4f' % (correct / total))
