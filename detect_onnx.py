"""
inferece on a dir where contains images
"""
import sys, os
sys.path.append('./')
from pathlib import Path
import onnx
import numpy as np
from PIL import Image
import onnxruntime
import argparse
from easydict import EasyDict as edict
import time
import logging

CFG = edict({
    'cate_names': [
        '/a/airfield', '/a/art_school', '/b/bridge', '/c/crosswalk', '/d/downtown',
        '/a/army_base', '/f/fire_station', '/h/highway', '/g/gas_station', '/c/church/outdoor'
    ]
})

def get_logger():
    logger = logging.getLogger()    # initialize logging class
    format = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(format)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    return logger

def build_onnx_sess(onnx_file):
    onnx_sess = onnxruntime.InferenceSession(onnx_file)
    input_names = [node.name for node in onnx_sess.get_inputs()]
    output_names = [node.name for node in onnx_sess.get_outputs()]
    logger.info(f'input_names: {input_names}')    
    logger.info(f'output_names: {output_names}')
    assert len(input_names) == 1
    assert len(output_names) == 1
    return onnx_sess, input_names[0], output_names[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='checkpoints/resnet18/model.onnx', help='onnx file')
    parser.add_argument('--dir', type=str, default='data/demo_images', help='images dir')
    args = parser.parse_args()
    
    #logger
    logger = get_logger()
    
    # check args
    if not Path(args.onnx).is_file():
        print(f'model file <{args.onnx}> do not exits')
    if not Path(args.onnx).is_dir():
        print(f'images dir <{args.dir}> do not exits')
    
    onnx_sess, inp_name, out_name = build_onnx_sess(args.onnx)

    img_dir = Path(args.dir)
    while True:
        jpg = list(img_dir.glob('*.jpg'))
        png = list(img_dir.glob('*.png'))
        images = [*jpg, *png]
        for img_file in images:
            img = Image.open(str(img_file)).convert('RGB').resize((224,224))
            img = np.array(img).astype(np.float32)
            image_tensor = img.transpose(2, 0, 1)
            image_tensor = image_tensor[np.newaxis, :]
            out = onnx_sess.run([out_name], input_feed={inp_name: image_tensor})
            scores = out[0]
            class_idx = np.argmax(scores)
            logger.info(f"{img_file.name}: {CFG.cate_names[class_idx]}")
            time.sleep(1)
