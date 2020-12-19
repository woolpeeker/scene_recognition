import sys, os
sys.path.append('./')
import argparse
import yaml
from easydict import EasyDict as edict
from PIL import Image
import torch.nn.functional as F
from model import Model, build_model
import torch.onnx
import torchvision.transforms as T
import networks, modules
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/resnet18.yaml', help='cfg file')

group = parser.add_mutually_exclusive_group()
group.add_argument('--eval', action='store_true', help='eval on widerface')
group.add_argument('--train', action='store_true', help='train')
group.add_argument('--demo', action='store_true', help='run a demo on an image')
group.add_argument('--onnx', type=str, help='onnx output path')

parser.add_argument('--image', '-i', type=str, help='demo input image')

args = parser.parse_args()

def run_train(args):
    cfg = edict(yaml.load(open(args.cfg), Loader=yaml.Loader))
    model = build_model(cfg, eval=False)
    ckpt_cb = ModelCheckpoint(
        os.path.join(cfg.ckpt, 'epoch{epoch}.pth'),
        save_last=True,
    )
    trainer = pl.Trainer(
        gpus=cfg.gpus,
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        checkpoint_callback=ckpt_cb
    )
    trainer.fit(model)

def run_eval(args):
    cfg = edict(yaml.load(open(args.cfg), Loader=yaml.Loader))
    model = build_model(cfg, eval=True)
    trainer = pl.Trainer(gpus=[0])
    trainer.test(model)

def run_demo(args):
    image = Image.open(args.image)
    cfg = edict(yaml.load(open(args.cfg), Loader=yaml.Loader))
    input_size = cfg.model.input_size
    model = build_model(cfg, eval=True)
    transform = T.Compose([T.Resize(input_size), T.CenterCrop(input_size), T.ToTensor()])
    input_image = transform(image)
    outputs = model(input_image)
    
    print(outputs)

def run_onnx(args):
    cfg = edict(yaml.load(open(args.cfg), Loader=yaml.Loader))
    input_size = cfg.model.input_size
    inp = torch.randn([1, 3, input_size, input_size])
    model = build_model(cfg, eval=True)
    torch.onnx.export(
        model,
        inp,
        args.onnx,
        opset_version=8,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    

if __name__ == '__main__':
    if args.train:
        run_train(args)
    elif args.eval:
        run_eval(args)
    elif args.demo:
        run_demo(args)
    elif args.onnx:
        run_onnx(args)
    else:
        raise RuntimeError('must select a function')
