
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

from dataloader import places_diy_v2, places_diy
import modules
from modules import NET_REGISTRY

def build_model(cfg, eval=False):
    Arch_type = ModelV2
    print(f"Arch_type: {Arch_type}")
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
            print('warning: ckpt file not found, %s' % ckpt_path)
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
            preprocess_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, -1, 1, 1])
            preprocess_std = torch.tensor([0.229, 0.224, 0.225]).reshape([1, -1, 1, 1])
            self.register_buffer('preprocess_mean', preprocess_mean)
            self.register_buffer('preprocess_std', preprocess_std)
            
    
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
            return (images - self.preprocess_mean) / self.preprocess_std
        else:
            return images / 127.5 - 1
    
    # ==============================================================================
    #                     training
    #===============================================================================
    def training_step(self, batch, batch_idx):
        return self.training_forward(batch)
    
    def training_forward(self, batched_inputs):
        images, labels = batched_inputs
        images = self.preprocessing(images)
        out_dict = self.net(images)
        logits = out_dict['outputs']
        
        BS = logits.shape[0]
        inout_mask = labels < 2
        label_mask = labels >= 2
        inout_logits = logits[inout_mask][:, :2]
        label_logits = logits[label_mask][:, 2:]
        gt_inout = labels[inout_mask]
        gt_labels = labels[label_mask] - 2
        inout_loss = F.cross_entropy(inout_logits, gt_inout, reduction='sum') / BS
        label_loss = F.cross_entropy(label_logits, gt_labels, reduction='sum') / BS
        loss = inout_loss + label_loss

        inout_acc = modules.accuracy(inout_logits, gt_inout, topk=(1, ))[0]
        label_top1, label_top5 = modules.accuracy(label_logits, gt_labels, topk=(1, 5))

        train_result = pl.TrainResult(minimize=loss)
        train_result.log_dict({
            'label_top1': label_top1,
            'label_top5': label_top5,
            'inout_acc': inout_acc
        }, prog_bar=True, logger=True, on_step=True)
        return train_result        
    
    def train_dataloader(self):
        loader = places_diy.get_places_diy_loader(
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
        logits = out_dict['outputs']

        inout_logits = logits[:, :2]
        inout_prob = F.softmax(inout_logits, dim=-1)
        label_logits = logits[:, 2:]
        label_prob = F.softmax(label_logits, dim=-1)
        return {
            'inout_prob': inout_prob,
            'label_prob': label_prob
        }

    # ==============================================================================
    #                     validate
    #===============================================================================
    def val_dataloader(self):
        loader = places_diy.get_places_diy_loader(
            size=self.hparams.input_size,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            mode='eval'
        )
        return loader
    
    def validation_step(self, batch, batch_idx):
        images, gt_classes = batch
        out_dict = self.inference_forward(images)
        result = pl.EvalResult()
        result.log_dict({
            'inout_prob': out_dict['inout_prob'],
            'label_prob': out_dict['label_prob'],
            'gt_classes': gt_classes
        })

        return result
    
    def validation_step_end(self, batch_parts):
        return batch_parts
    
    def validation_epoch_end(self, validation_step_outputs):
        gt_classes = validation_step_outputs.gt_classes
        inout_mask = gt_classes < 2
        gt_inout = gt_classes[inout_mask]
        inout_prob = validation_step_outputs.inout_prob[inout_mask]
        inout_acc = modules.accuracy(inout_prob, gt_inout, topk=(1,))[0]

        label_mask = gt_classes >= 2
        label_prob = validation_step_outputs.label_prob[label_mask]
        gt_label = gt_classes[label_mask] - 2
        top1, top5 = modules.accuracy(label_prob, gt_label, (1, 5))
        cate_topk = modules.accuracy_per_cate(label_prob, gt_label, (1, 5))
        result = pl.EvalResult()
        result.log_dict({
            'top1': top1.item(),
            'top5': top5.item(),
            'inout_acc': inout_acc.item(),
        }, prog_bar=True, on_epoch=True)
        print('cate_topk', cate_topk)
        print('top1: ', top1.item())
        print('top5: ', top5.item())
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

class ModelV2(Model):
    def train_dataloader(self):
        loader = places_diy_v2.get_places_diy_v2_loader(
            size=self.hparams.input_size,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            mode='train'
        )
        return loader
    
    def val_dataloader(self):
        loader = places_diy_v2.get_places_diy_v2_loader(
            size=self.hparams.input_size,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            mode='eval'
        )
        return loader