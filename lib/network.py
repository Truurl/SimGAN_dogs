import pytorch_lightning as L
import config as cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from pathlib import Path
import h5py
from PIL import Image
import numpy as np
from .image_history_buffer import ImageHistoryBuffer
from .image_utils import generate_avg_histograms, generate_img_batch, calc_acc
import wandb
import os
import matplotlib.pyplot as plt

class Refiner(nn.Module):

    def __init__(self, input_features, nb_features=64):
        super().__init__()

        def resnet_block(input_features, nb_features):
            layers = [
                nn.Conv2d(input_features, nb_features, 3, 1, 1),
                # nn.BatchNorm2d(nb_features),
                # nn.LeakyReLU(),
                nn.ReLU(),
                nn.Conv2d(nb_features, nb_features, 3, 1, 1),
                # nn.BatchNorm2d(nb_features),
                # nn.LeakyReLU() 
                nn.ReLU(),
            ]

            return layers

        self.model = nn.Sequential(
            nn.Conv2d(input_features, nb_features, 3, stride=1, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(nb_features),
            *resnet_block(nb_features, nb_features),
            *resnet_block(nb_features, nb_features),
            *resnet_block(nb_features, nb_features),
            *resnet_block(nb_features, nb_features),
            nn.Conv2d(nb_features, input_features, 1, 1, 0),
            # nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_features, 96, 3, 2, 1),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(96),

            nn.Conv2d(96, 64, 3, 2, 1),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(64),

            nn.MaxPool2d(3, 2, 1),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 1, 1, 0),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(32),

            nn.Conv2d(32, 2, 1, 1, 0),
            nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.BatchNorm2d(2),

            # nn.Softmax()
        )

    def forward(self, x):
        convs = self.convs(x)
        output = convs.view(convs.size(0), -1, 2)
        return output

class SimGAN(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        lr_d: float = cfg.d_lr,
        lr_r: float = cfg.r_lr,
        batch_size: int = cfg.batch_size,
        r_step_size: int = cfg.r_step_size,
        r_gamma: float = cfg.r_gamma,
        d_step_size: int = cfg.d_step_size,
        d_gamma: float = cfg.d_gamma,
        r_pretrain: int = cfg.r_pretrain,
        d_pretrain: int = cfg.d_pretrain,
        k_d: int = cfg.k_d,
        k_r: int= cfg.k_r,
        delta: float = cfg.delta,
        buffer_size: int = cfg.buffer_size,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.automatic_optimization = False
        # networks
        input_features = (channels, width, height)
        self.refiner = Refiner(channels, nb_features=64)
        self.discriminator = Discriminator(channels)

        self.r_pretrain = self.hparams.r_pretrain
        self.d_pretrain = self.hparams.d_pretrain

        self.k_r = self.hparams.k_r
        self.k_d = self.hparams.k_d

        self.training_steps = 0
        self.d_pretrain_steps = 0
        self.r_pretrain_steps = 0

        self.discriminator_out_size = self.discriminator(torch.rand((cfg.batch_size, cfg.img_channels, cfg.img_height, cfg.img_width))).size()
        
        self.image_history_buffer = ImageHistoryBuffer((0, channels, height, width),
                                                        buffer_size * 10, batch_size)
        
        self.val_synth_batch = None
        self.val_real_batch = None

        self.automatic_optimizer = True
        self.metrics = { "training_step": 0,
                            "refiner": {
                                "loss": {
                                    "adv": 0,
                                    "l1reg": 0,
                                        "total": 0,
                                },
                                "accuracy"  : 0,
                            },
                                "discriminator": {
                                    "loss": {
                                        "real": 0,
                                        "ref": 0,
                                        "total": 0,
                                    },
                                    "accuracy": {
                                        "real": 0,
                                        "ref": 0,
                                    },
                                },
                       }

        if not os.path.exists(f'{cfg.train_res_path}/models'):
            os.makedirs(f'{cfg.train_res_path}/models')


    def forward(self, x):
        return self.refiner(x)
    
    def adversarial_loss(self, x, y):
        return F.binary_cross_entropy_with_logits(x, y, reduction='sum')
    
    def refiner_loss(self, x, y):
        return self.hparams.delta * F.l1_loss(x, y, reduction='sum')

    def refiner_step(self, batch, optimizer=None):

        if self.r_pretrain != self.r_pretrain_steps:

            synth_images, synth_target = batch['synth']
            refined_images = self.refiner(synth_images)

            r_loss = self.refiner_loss(refined_images, synth_images)
            self.r_pretrain_steps =+ 1

            return r_loss

        elif self.k_r:

            synth_images, synth_target = batch['synth']
            refined_images = self.refiner(synth_images) 
         
            d_ref_pred = self.discriminator(refined_images).view(-1, 2)
            d_real_y = torch.zeros(d_ref_pred.size(), dtype=torch.float).to(self.device)
            d_real_y[:, 0] = 1.
         
            acc_adv = calc_acc(d_ref_pred, 'real')
         
            r_loss_reg = self.refiner_loss(synth_images, refined_images)
            r_loss_adv = self.adversarial_loss(d_ref_pred, d_real_y)
                
            r_loss = r_loss_reg + r_loss_adv
         
            self.metrics['refiner']['loss']['total'] += r_loss / self.hparams_batch_size
            self.metrics['refiner']['loss']['adv'] += r_loss_reg / self.hparams_batch_size
            self.metrics['refiner']['loss']['l1reg'] += r_loss / self.hparams_batch_size
            self.metrics['refiner']['accuracy'] += acc_adv

            # loss_dict = {'r_loss': r_loss}

            self.k_r -= 1

        return r_loss

    def disctiminator_step(self, batch, optimizer=None):

        if self.d_pretrain != self.d_pretrain_steps:

            synth_batch = batch['synth'][0]
            real_batch = batch['real'][0]

            d_real_pred = self.discriminator(real_batch).view(-1, 2)
            d_real_y = torch.zeros(d_real_pred.size(), dtype=torch.float).to(self.device)
            d_real_y[:, 0] = 1.

            d_loss_real = self.adversarial_loss(d_real_pred, d_real_y)

            refined_batch = self.refiner(synth_batch)

            d_ref_pred = self.discriminator(refined_batch).view(-1, 2)
            d_ref_y = torch.ones(d_ref_pred.size(), dtype=torch.float).to(self.device)
            d_ref_y[:, 1] = 1.
            
            d_loss_ref = self.adversarial_loss(d_ref_pred, d_ref_y)

            d_loss = d_loss_real + d_loss_ref

            self.d_pretrain_steps =+ 1

            return d_loss

        elif self.k_d:

            synth_batch = batch['synth'][0]
            real_batch = batch['real'][0]
                    
            refined_batch = self.refiner(synth_batch)

            half_batch_from_image_history = self.image_history_buffer.get_from_image_history_buffer()
            self.image_history_buffer.add_to_image_history_buffer(refined_batch.cpu().data.numpy().transpose((0, 1, 3, 2)))

            if len(half_batch_from_image_history):
                torch_type = torch.from_numpy(half_batch_from_image_history).transpose(2,3)
                # v_type = Variable(torch_type).cuda(cfg.cuda_num).transpose(2, 3)
                refined_batch[:cfg.batch_size // 2] = torch_type

            d_real_pred = self.discriminator(real_batch).view(-1, 2)
            d_real_y = torch.zeros(d_real_pred.size(), dtype=torch.float).to(self.device)
            d_real_y[:, 0] = 1.
            
            d_loss_real = self.adversarial_loss(d_real_pred, d_real_y)
            
            d_ref_pred = self.discriminator(refined_batch).view(-1, 2)
            d_ref_y = torch.ones(d_ref_pred.size(), dtype=torch.float).to(self.device)
            d_ref_y[:, 1] = 1.
            
            d_loss_ref = self.adversarial_loss(d_ref_pred, d_ref_y)
            d_loss = d_loss_real + d_loss_ref

            acc_real = calc_acc(d_real_pred, 'real')
            acc_ref = calc_acc(d_ref_pred, 'refine')
         
            self.metrics['discriminator']['loss']['total'] += d_loss / self.hparams_batch_size
            self.metrics['discriminator']['loss']['real'] += d_loss_real / self.hparams_batch_size
            self.metrics['discriminator']['loss']['ref'] += d_loss_ref / self.hparams_batch_size
            self.metrics['discriminator']['accuracy']['real'] += acc_real
            self.metrics['discriminator']['accuracy']['real'] += acc_ref
            
            self.k_d -= 1
        
        # self.manual_backwards(d_loss)
        # optimizer_d.step()
        # optimizer_d.zero_grad()
        # self.untoggle_optimizer(optimizer_d)
        # self.log("d_loss", d_loss, prog_bar=True)

        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        if optimizer_idx == 0:
            loss = self.refiner_step(batch, optimizer=optimizer_idx)
        
        if optimizer_idx == 1:
            loss = self.disctiminator_step(batch, optimizer=optimizer_idx)
        
        if self.k_r == 0 and self.k_d == 0:
            self.k_r = self.hparams.k_r
            self.k_d = self.hparams.k_d
        
        print(loss)
        return loss

    # def training_epoch_end(self, outputs):

    def validation_step(self, batch, batch_idx):

        torch.save(self.discriminator.state_dict(), f'{cfg.train_res_path}/{cfg.D_path}' % self.training_steps)
        torch.save(self.refiner.state_dict(), f'{cfg.train_res_path}/{cfg.R_path}' % self.training_steps)

        synth_images = batch['synth'][0]
        real_images = batch['real'][0]

        if self.val_synth_batch == None:
            self.val_synth_batch = synth_images

        if self.val_real_batch == None:
            self.val_real_batch = real_images

        # d_real_pred = self.discriminator(real_images).view(-1, 2)
        # d_real_y = torch.zeros(d_real_pred.size(0), dtype=torch.long, device=self.device)

        # d_real_loss = self.adversarial_loss(d_real_pred, d_real_y)

        # ref_imgs = self.refiner(synth_images)
        # ref_loss = self.refiner_loss(synth_images, ref_imgs)

        # d_synth_pred = self.discriminator(synth_images).view(-1, 2)
        # d_synth_y = torch.ones(d_synth_pred.size(0), dtype=torch.long, device=self.device)

        # d_synth_loss = self.adversarial_loss(d_synth_pred, d_synth_y)

        # d_loss = d_real_loss + d_synth_loss



        # self.log('d_loss', d_loss.item()/2)
        # self.log('d_real_loss', d_real_loss)
        # self.log('d_refine_loss', d_synth_loss)
        # self.log('ref_loss', ref_loss)
        # self.log('acc_real', real_acc)
        # self.log('acc_ref', ref_acc)
        if batch_idx == 0:

            self.metrics['training_step'] = self.training_steps

            self._avg_metrics()

            # wandb.log(self.metrics)

            # self._zero_metrics()

            self.training_steps += 1

    def configure_optimizers(self):
        lr_r = self.hparams.lr_r
        lr_d = self.hparams.lr_d

        r_step_size = self.hparams.r_step_size
        r_gamma = self.hparams.r_gamma
        d_step_size = self.hparams.d_step_size
        d_gamma = self.hparams.d_gamma

        # optimizer_r = torch.optim.Adam(self.refiner.parameters(), lr=lr_r)
        optimizer_r = torch.optim.SGD(self.refiner.parameters(), lr=lr_r)
        optimizer_d = torch.optim.SGD(self.discriminator.parameters(), lr=lr_d)

        return [optimizer_r, optimizer_d], []
    
        r_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_r, step_size=r_step_size, gamma=r_gamma)
        d_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=d_step_size, gamma=d_gamma)
        
        return [optimizer_r, optimizer_d], [r_lr_scheduler, d_lr_scheduler]
    
    def  generate_batch_train_image(self, syn_image_batch, ref_image_batch, real_image_batch, step_index=-1):

        pic_path = os.path.join(cfg.train_res_path, 'step_%d.png' % step_index)
        hist_path = os.path.join(cfg.train_res_path, 'step_%d_hist.png' % step_index)
        img = generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, pic_path)
        hist = generate_avg_histograms(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, hist_path)
        # generate_img_batch
        img.save(pic_path, 'png')
        hist.savefig(hist_path)
        self.logger.experimental.log({f"step_{step_index}": wandb.Image(img), f"step_{step_index}_hist": wandb.Image(hist)})
        # self.log(f"step_{step_index}", wandb.Image(img))
        # self.log(f"step_{step_index}_hist", wandb.Image(hist))
        # wandb.log({f"step_{step_index}": wandb.Image(img), f"step_{step_index}_hist": wandb.Image(hist)})
        plt.close(hist)

    def _get_training_steps(self):
        return self.training_steps
    
    def _get_static_val_batch(self):
        return (self.val_synth_batch, self.val_real_batch)
    
    def _avg_metrics(self):
            
            for key, value in self.metrics['refiner']['loss'].items():
                mean = value / self.k_r
                value = mean
            
            mean = self.metrics['refiner']['accuracy'] / self.k_d
            self.metrics['refiner']['accuracy'] = mean

            for key, value in self.metrics['discriminator']['loss'].items():
                mean = value / self.k_d
                value = mean
            
            for key, value in self.metrics['discriminator']['accuracy'].items():
                mean = value / self.k_d
                value = mean

    def _reset_metrics(self):
            for key, value in self.metrics['refiner']['loss'].items():
                value = 0
            
            self.metrics['refiner']['accuracy'] = 0

            for key, value in self.metrics['discriminator']['loss'].items():
                value = 0
            
            for key, value in self.metrics['discriminator']['accuracy'].items():
                value = 0

    def _get_metrics(self):
        
        if self.training_steps > 0:
            for key, value in self.metrics['refiner']['loss'].items():
                mean = value / self.k_r
                value = mean
            
            mean = self.metrics['refiner']['accuracy'] / self.k_d
            self.metrics['refiner']['accuracy'] = mean

            for key, value in self.metrics['discriminator']['loss'].items():
                mean = value / self.k_d
                value = mean
            
            for key, value in self.metrics['discriminator']['accuracy'].items():
                mean = value / self.k_d
                value = mean
                
            ret = self.metrics

            for key, value in self.metrics['refiner']['loss'].items():
                value = 0
            
            self.metrics['refiner']['accuracy'] = 0

            for key, value in self.metrics['discriminator']['loss'].items():
                value = 0
            
            for key, value in self.metrics['discriminator']['accuracy'].items():
                value = 0
                
            return ret

        return None
