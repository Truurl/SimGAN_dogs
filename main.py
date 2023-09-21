import wandb
from PIL import Image
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import config as cfg
from lib.network import SimGAN, Discriminator, Refiner
from lib.data_module import EyeDatasetModule
from lib.image_utils import generate_avg_histograms, generate_img_batch
import wandb
import matplotlib.pyplot as plt
import os

class ImageRefinerSamplesCallback(L.callbacks.Callback):

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        def  generate_batch_train_image(syn_image_batch, ref_image_batch, real_image_batch, step_index=-1):

            pic_path = os.path.join(cfg.train_res_path, 'step_%d.png' % step_index) 
            hist_path = os.path.join(cfg.train_res_path, 'step_%d_hist.png' % step_index)
            img = generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, 
                                     real_image_batch.cpu().data, pic_path)
            # hist, kl_score = generate_avg_histograms(syn_image_batch.cpu().data, 
                                                    #  ref_image_batch.cpu().data, 
                                                    #  real_image_batch.cpu().data, hist_path)

            
            trainer.logger.experiment.log({f'step.png': wandb.Image(img)})
            # trainer.logger.experiment.log(kl_score)
            # trainer.logger.experiment.log({f'step_hist.png': wandb.Image(hist)})
            
            # plt.close(hist)
            
        if batch_idx == 0:

            synth_images, real_images = pl_module._get_static_val_batch()
            ref_images = pl_module(synth_images)
            # synth_images = batch['synth'][0]
            # real_images = batch['real'][0]
            metrics_dict = pl_module._get_metrics()
            # print(metrics_dict)
            if metrics_dict:
                # print("senfing logs")
                trainer.logger.experiment.log(metrics_dict)
                pl_module._reset_metrics()
            
            generate_batch_train_image(synth_images, ref_images, real_images, 
                                       step_index=pl_module._get_training_steps())
            

if __name__ == '__main__':
    
    wandb_logger = L.loggers.WandbLogger(project="simgan_lightning")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((cfg.img_width, cfg.img_height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])

    dm = EyeDatasetModule(cfg.syn_path,
                           cfg.syn_datasets,
                           cfg.real_path,
                           cfg.real_datesets,
                           batch_size=cfg.batch_size,
                           transform=transform
                           )
    model = SimGAN(cfg.img_channels, cfg.img_width, cfg.img_height)
    trainer = L.Trainer(
        accelerator="gpu",
        strategy='ddp',
        devices = -1,
        precision=16,
        max_epochs = cfg.train_steps,
        check_val_every_n_epoch = 1,
        logger=wandb_logger,
        callbacks = [ImageRefinerSamplesCallback()]

    )

    trainer.fit(model, dm)
    print('done')
