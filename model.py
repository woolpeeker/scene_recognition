
import os
import json
import copy
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

import pytorch_lightning as pl
from easydict import EasyDict as edict

from dataloader import places
import modules
from modules import NET_REGISTRY

def build_model(cfg, eval=False):
    Arch_type = Model
    hparams = Arch_type.default_hparams()
    for k, v in cfg.model.items():
        assert k in hparams
        hparams[k] = v

    if not eval:
        model = Arch_type(hparams)
    else:
        ckpt_path = os.path.join(cfg.ckpt, 'last.ckpt')
        if os.path.exists(ckpt_path):
            model = Arch_type.load_from_checkpoint(ckpt_path)
            model.hparams = hparams
        else:
            model = Arch_type(hparams)
        model = model.eval()
    return model

class Model(pl.LightningModule):
    @classmethod
    def default_hparams(cls):
        hparams = edict()
        hparams.input_size             = 224
        hparams.net_name               = None
        hparams.net_build_args         = None
        hparams.net_forward_args       = None
        hparams.preprocess             = 'ImageNet'
        # training
        hparams.loss_alpha = 0.4
        hparams.lr = 1e-3
        hparams.lr_decays = [0.1, 1, 0.2, 0.4]
        hparams.lr_decays_epochs = [1, 5, 10]
        hparams.batch_size = 64
        hparams.num_workers = 6
        return hparams
    
    def __init__(self, hparams, name='Model'):
        super().__init__()
        super().__init__()
        if hparams is None:
            hparams = self.default_hparams()
        else:
            hparams = copy.deepcopy(hparams)
        hparams = edict(hparams)
        self.hparams = hparams
        self.name = name
        if 'net_args' in hparams is not None: # historical legacy
            hparams.net_build_args = hparams.net_args
        if 'net_build_args' in hparams and hparams.net_build_args is not None:
            self.net = NET_REGISTRY.get(hparams.net_name)(**hparams.net_build_args)
        else:
            self.net = NET_REGISTRY.get(hparams.net_name)()
        
        if self.hparams.preprocess == 'ImageNet':
            self.preprocess_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, -1, 1, 1])
            self.preprocess_std = [0.229, 0.224, 0.225]
    
    def hparams_sanity(self, hparams):
        lr_decays_epochs = hparams.lr_decays_epochs
        assert len(lr_decays_epochs) == len(hparams.lr_decays) - 1
        for i in range(len(lr_decays_epochs)-1):
            assert lr_decays_epochs[i] <= lr_decays_epochs[i+1]
    
    def forward(self, batched_inputs):
        if self.training:
            return self.training_forward(batched_inputs)
        else:
            return self.inference_forward(batched_inputs)
    
    def preprocessing(self, images):
        # images: [B, 3, H, W]
        if self.hparams.preprocess == 'ImageNet':
            mean = torch.tensor(self.preprocess_mean).to(images.device).reshape([1, -1, 1, 1])
            std = torch.tensor(self.preprocess_std).to(images.device).reshape([1, -1, 1, 1])
            return (images - mean) / std
        else:
            return images / 127.5 - 1
    
    # ==============================================================================
    #                     training
    #===============================================================================
    def training_step(self, batch, batch_idx):
        return self.training_forward(batch)
    
    def training_forward(self, batched_inputs):
        images, gt_classes = batched_inputs
        images = self.preprocessing(images)
        out_dict = self.net(images)
        loss = F.cross_entropy(out_dict['outputs'], gt_classes)
        top1, top5 = modules.accuracy(out_dict['outputs'], gt_classes, topk=(1, 5))
        train_result = pl.TrainResult(minimize=loss)
        train_result.log_dict({
            'top1': top1,
            'top5': top5,
        }, prog_bar=True, logger=True, on_step=True)
        return train_result
    
    def train_dataloader(self):
        loader = places.get_places_10_loader(
            size=self.hparams.input_size,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            mode='train'
        )
        return loader
    
    def configure_optimizers(self):
        decays = self.hparams.lr_decays
        decay_epochs = self.hparams.lr_decays_epochs
        adam = optim.Adam(self.parameters(), lr=float(self.hparams.lr))
        def lr_fn(epoch):
            for i in range(len(decay_epochs)):
                if decay_epochs[i] > epoch:
                    break
            return decays[i]
        lr_scheduler = optim.lr_scheduler.LambdaLR(adam, lr_fn)
        return [adam], [lr_scheduler]
    
    # ==============================================================================
    #                     inference
    #===============================================================================
    def inference_forward(self, batched_inputs):
        images = batched_inputs
        images = self.preprocessing(images)
        out_dict = self.net(images)
        outputs = out_dict['outputs']
        outputs = F.softmax(outputs, 1)
        return {
            'pred_logits': out_dict['outputs'],
        }

    # ==============================================================================
    #                     validate
    #===============================================================================
    def val_dataloader(self):
        loader = places.get_places_10_loader(
            size=self.hparams.input_size,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            mode='eval'
        )
        return loader
    
    def validation_step(self, batch, batch_idx):
        images, gt_classes = batch
        out_dict = self.inference_forward(images)
        pred_prob = F.softmax(out_dict['pred_logits'], 1)
        result = pl.EvalResult()
        result.log_dict({
            'pred_prob': pred_prob,
            'gt_classes': gt_classes
        })

        return result
    
    def validation_step_end(self, batch_parts):
        if isinstance(batch_parts.top1, list):
            result = pl.EvalResult()
            result.log_dict({
                'pred_prob': torch.cat(batch_parts.pred_prob),
                'gt_classes': torch.cat(batch_parts.gt_classes)
            })
        else:
            result = batch_parts
        return result
    
    def validation_epoch_end(self, validation_step_outputs):
        pred_prob = validation_step_outputs.pred_prob
        gt_classes = validation_step_outputs.gt_classes
        top1, top5 = modules.accuracy(pred_prob, gt_classes, (1, 5))
        cate_topk = modules.accuracy_per_cate(pred_prob, gt_classes, (1, 5))
        result = pl.EvalResult()
        result.log_dict({
            'top1': top1,
            'top5': top5,
        }, prog_bar=True, on_epoch=True)
        print('cate_topk', cate_topk)
        return result

    # ==============================================================================
    #                     test
    #===============================================================================
    def test_dataloader(self):
        return self.val_dataloader()
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_step_end(self, *args, **kwargs):
        return self.validation_step_end(*args, **kwargs)
    
    def test_epoch_end(self, validation_step_outputs):
        return self.validation_epoch_end(validation_step_outputs)