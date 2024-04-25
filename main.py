import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from itertools import cycle

from lib.dateset import HDF5Dataset, CacheHDF5Dataset, MemHDF5Dataset, MemMapDataset
from lib.image_history_buffer import ImageHistoryBuffer
from lib.network import Discriminator, Refiner, InceptionV3
from lib.image_utils import generate_img_batch, generate_avg_histograms, calc_acc
from lib.metrics import calculate_activation_statistics, calculate_frechet_distance, calculate_fretchet, calculate_kl_score
import config as cfg
import os
import matplotlib.pyplot as plt
import logging
import wandb
import numpy as np
import h5py
from PIL import Image
import signal, sys

from queue import Queue
import threading

import copy

def terminate_signal(signalnum, handler):
    print ('Terminate the process')
    # save results, whatever...
    wandb.finish()
    sys.exit()

class Main(object):
    def __init__(self):
        # network
        self.R = None
        self.D = None
        self.opt_R = None
        self.opt_D = None
        self.self_regularization_loss = None
        self.local_adversarial_loss = None
        self.delta = None

        # data
        self.syn_train_loader = None
        self.real_loader = None
        # iter
        self.syn_train_iter = None
        self.real_image_iter = None
        # learning rate scheduler
        self.d_lr_scheduler = None
        self.r_lr_scheduler = None

        self.metrics_queue = Queue()

        if not os.path.exists(f'{cfg.train_res_path}/models'):
            os.makedirs(f'{cfg.train_res_path}/models')
        # os.mkdir(f'cfg.train_res_path/models')
        logging.basicConfig(filename=f'{cfg.train_res_path}/training_progress.log', level=logging.INFO)
        self.run = wandb.init(
            # set the wandb project where this run will be logged
            project="simgan",
            # track hyperparameters and run metadata
            config={
                'd_lr': cfg.d_lr,
                'r_lr': cfg.r_lr,
                'delta': cfg.delta,
                'r_step_size': cfg.r_step_size,
                'r_gamma': cfg.r_gamma,
                'd_step_size': cfg.d_step_size,
                'd_gamma': cfg.d_gamma,
                'img_width': cfg.img_width,
                'img_height': cfg.img_height,
                'img_channels': cfg.img_channels,
                'r_pretrain': cfg.r_pretrain,
                'd_pretrain': cfg.d_pretrain,
                'train_steps': cfg.train_steps,
                'batch_size': cfg.batch_size,
                'buffer_size': cfg.buffer_size,
                'k_d': cfg.k_d,
                'k_r': cfg.k_r,
                "nb_features": cfg.nb_features,
                "refiner_arch":
                {
                    "n_resnets": cfg.n_resnets,
                    "n_heads": cfg.n_heads
                }
            },
            name=f'2-head-discriminator'
        )

    def get_next_synth_batch(self):
        try:
            syn_image_batch  = next(self.syn_train_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.syn_train_iter = iter(self.syn_train_loader)
            syn_image_batch = next(self.syn_train_iter)

        syn_image_batch = syn_image_batch.to(cfg.dev, non_blocking=True)
        return syn_image_batch

    def get_next_real_batch(self):
        try:
            real_image_batch = next(self.real_image_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.real_image_iter = iter(self.real_loader)
            real_image_batch  = next(self.real_image_iter)

        real_image_batch = real_image_batch.to(cfg.dev, non_blocking=True)
        return real_image_batch

    def get_next_synth_seg_batch(self):
        try:
            syn_seg_image_batch  = next(self.syn_seg_train_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.syn_train_iter = iter(self.syn_seg_train_loader)
            syn_seg_image_batch = next(self.syn_seg_train_iter)

        syn_seg_image_batch = syn_seg_image_batch.to(cfg.dev, non_blocking=True)
        return syn_seg_image_batch

    def get_next_real_seg_batch(self):
        try:
            real_seg_image_batch = next(self.real_seg_image_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.real_seg_image_iter = iter(self.real_seg_loader)
            real_seg_image_batch  = next(self.real_seg_image_iter)

        real_seg_image_batch = real_seg_image_batch.to(cfg.dev, non_blocking=True)
        return real_seg_image_batch
    
    def reset_iters(self):
        self.real_image_iter = iter(self.real_loader)
        self.syn_train_iter = iter(self.syn_train_loader)

    def reset_seg_iters(self):
        self.real_seg_image_iter = iter(self.real_seg_loader)
        self.syn_seg_train_iter = iter(self.syn_seg_train_loader)

    def metrics_thread(self):

        while True:

            data = self.metrics_queue.get()

            if data != None:

                step = data['step']
                l1_dict = data['l1_dict']
                # val_synth_batch = data['val_synth_batch']
                val_ref_batch = data['val_ref_batch']
                # val_real_batch = data['val_real_batch']

                if step % cfg.save_per == 0:

                    self.generate_batch_train_image(self.val_synth_batch,
                                                    val_ref_batch,
                                                    self.val_real_batch,
                                                    step_index=step)
                    kl_score = calculate_kl_score(self.val_synth_batch.cpu().data,
                                                val_ref_batch.cpu().data,
                                                self.val_real_batch.cpu().data)
                    if cfg.fretchet:
                        fretchet_dist_real_ref = calculate_fretchet(self.val_real_batch,
                                                                val_ref_batch,
                                                                self.inception_model)
                        fretchet_dist_synth_ref = calculate_fretchet(self.val_synth_batch,
                                                                val_ref_batch,
                                                                self.inception_model)
                        fretchet_score = {
                            "fretcher" :{
                                'real-ref': fretchet_dist_real_ref,
                                'synth-ref': fretchet_dist_synth_ref,
                            }
                        }
                    else:
                        fretchet_score = {}
                    log_dict = kl_score | l1_dict | fretchet_score
                else:
                    log_dict = l1_dict
                wandb.log(log_dict)


    def get_next_batches(self):
        real_batch = self.get_next_real_batch()
        synth_batch = self.get_next_synth_batch()

        if real_batch.size(0) != synth_batch.size(0):
            self.reset_iters()
            real_batch = self.get_next_real_batch()
            synth_batch = self.get_next_synth_batch()

        return (real_batch, synth_batch)

    def get_next_seg_batches(self):
        real_seg_batch = self.get_next_real_batch()
        synth_seg_batch = self.get_next_synth_batch()

        if real_seg_batch.size(0) != synth_seg_batch.size(0):
            self.reset_iters()
            real_seg_batch = self.get_next_real_batch()
            synth_seg_batch = self.get_next_synth_batch()

        return (real_seg_batch, synth_seg_batch)
    
    def build_network(self):
        logging.info('=' * 50)
        logging.info('Building network...')
        self.R = Refiner(cfg.n_resnets, cfg.img_channels,
                         nb_features=cfg.nb_features,
                         num_heads=cfg.n_heads)
        self.D = Discriminator(input_features=cfg.img_channels)
        self.D_seg = Discriminator(input_features=cfg.img_channels)


        if cfg.fretchet:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception_model = InceptionV3([block_idx])

        if cfg.cuda_use:
            self.R.cuda(cfg.cuda_num)
            self.D.cuda(cfg.cuda_num)
            self.D_seg.cuda(cfg.cuda_num)
            if cfg.fretchet:
                self.inception_model.cuda(cfg.cuda_num)

        # self.opt_R = torch.optim.SGD(self.R.parameters(), lr=cfg.r_lr)
        # self.opt_D = torch.optim.SGD(self.D.parameters(), lr=cfg.d_lr)

        self.opt_R = torch.optim.Adam(self.R.parameters(), lr=cfg.r_lr)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr)
        self.opt_D_seg = torch.optim.Adam(self.D_seg.parameters(), lr=cfg.d_lr)
        # self.self_regularization_loss = nn.L1Loss(size_average=False)
        # self.local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)
        self.self_regularization_loss = nn.L1Loss(reduction='sum')
        self.local_adversarial_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.delta = cfg.delta
        self.r_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_R,
                                    step_size=cfg.r_step_size,
                                    gamma=cfg.r_gamma)
        self.d_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_D,
                                    step_size=cfg.d_step_size,
                                    gamma=cfg.d_gamma)

        self.val_real_batch = None
        self.val_synth_batch = None

        self.d_output_size = self.D(torch.rand((cfg.batch_size, cfg.img_channels, cfg.img_height, cfg.img_width), device=cfg.dev)).size()

    def print_training_setup(self):
        print('*** DATA ***')
        print(f'transform: {self.transform}')
        print(f'self.syn_train_loader: len: {len(self.syn_train_loader)} {self.syn_train_loader}')
        print(f'self.real_loader: len: {len(self.real_loader)} {self.real_loader}')
        print(f'self.syn_train_loader: len: {len(self.syn_seg_train_loader)} {self.syn_seg_train_loader}')
        print(f'self.real_loader: len: {len(self.real_seg_loader)} {self.real_seg_loader}')
        print('*** OPTS AND LOSSES')
        print(f'self.opt_R: {self.opt_R}')
        print(f'self.opt_D: {self.opt_D}')
        print(f'self.opt_D: {self.opt_D_seg}')
        print(f'self.self_regularization_loss: {self.self_regularization_loss}')
        print(f'self.local_adversarial_loss: {self.local_adversarial_loss}')
        print('*** MODELS ***')
        print(f'self.R\n{self.R}')
        print("\n\n")
        print(f'self.D\n{self.D}')
        print("\n\n")
        print(f'self.D\n{self.D_seg}')


    def load_data(self):
        logging.info('=' * 50)
        logging.info('Loading data...')
        self.transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize((cfg.img_width, cfg.img_height), antialias=True),
            transforms.Normalize(cfg.img_channels * (0.5,), cfg.img_channels * (0.5,))
            ])

        # syn_train_folder = torchvision.datasets.ImageFolder(root=cfg.syn_path, transform=transform)
        syn_train_folder = HDF5Dataset(cfg.syn_path, cfg.syn_datasets, self.transform)
        # syn_train_folder = MemMapDataset(cfg.syn_path, cfg.syn_datasets, self.transform, dataset_size=10000)
        self.syn_train_loader = Data.DataLoader(syn_train_folder, batch_size=cfg.batch_size,
                                                shuffle=True, pin_memory=True, num_workers=4)
        self.syn_train_iter = iter(self.syn_train_loader)
        logging.info('syn_train_batch %d' % len(self.syn_train_loader))

        syn_seg_train_folder = HDF5Dataset(cfg.syn_seg_path, cfg.syn_seg_datasets, self.transform)
        self.syn_seg_train_loader = Data.DataLoader(syn_seg_train_folder, batch_size=cfg.batch_size,
                                                shuffle=True, pin_memory=True, num_workers=4)
        self.syn_seg_train_iter = iter(self.syn_seg_train_loader)
        logging.info('syn_seg_train_batch %d' % len(self.syn_train_loader))

        # real_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transform)
        real_folder = HDF5Dataset(cfg.real_path, cfg.real_datasets, self.transform)
        # real_folder = MemMapDataset(cfg.real_path, cfg.real_datasets, self.transform, dataset_size=10000)
        # real_folder.imgs = real_folder.imgs[:2000]
        self.real_loader = Data.DataLoader(real_folder, batch_size=cfg.batch_size,
                                           shuffle=True, pin_memory=True, num_workers=4)
        self.real_image_iter = iter(self.real_loader)
        logging.info('real_batch %d' % len(self.real_loader))

        
        real_seg_folder = HDF5Dataset(cfg.real_seg_path, cfg.real_datasets, self.transform)
        self.real_seg_loader = Data.DataLoader(real_seg_folder, batch_size=cfg.batch_size,
                                           shuffle=True, pin_memory=True, num_workers=4)
        self.real_seg_image_iter = iter(self.real_seg_loader)
        logging.info('real_batch %d' % len(self.real_seg_loader))

    def pre_train_r(self):
        logging.info('=' * 50)
        if cfg.ref_pre_path:
            logging.info('Loading R_pre from %s' % cfg.ref_pre_path)
            self.R.load_state_dict(torch.load(cfg.ref_pre_path))
            return

        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        logging.info('pre-training the refiner network %d times...' % cfg.r_pretrain)

        for index in range(cfg.r_pretrain):
            self.opt_R.zero_grad()

            syn_image_batch = self.get_next_synth_batch()

            self.R.train()
            ref_image_batch = self.R(syn_image_batch)

            r_loss = self.self_regularization_loss(ref_image_batch, syn_image_batch)
            r_loss = torch.mul(r_loss, self.delta)

            r_loss.backward()
            self.opt_R.step()


    def pre_train_d(self):

        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        logging.info('pre-training the discriminator network %d times...' % cfg.r_pretrain)
        self.D.train()
        self.D_seg.train()
        self.R.eval()

        for index in range(cfg.d_pretrain):
            self.opt_D.zero_grad()

            real_image_batch, syn_image_batch = self.get_next_batches()

            # ============ ref image D ====================================================
            ref_image_batch = self.R(syn_image_batch)
            d_ref_pred = self.D(ref_image_batch).view(-1,2)
            d_ref_y = torch.zeros(d_ref_pred.size(), dtype=torch.float, device=cfg.dev)
            d_ref_y[:, 1] = 1

            acc_ref = calc_acc(d_ref_pred, 'refine')
            d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
            d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)

            # ============ real image D ====================================================
            d_real_pred = self.D(real_image_batch).view(-1,2)
            d_real_y = torch.zeros(d_real_pred.size(), dtype=torch.float, device=cfg.dev)
            d_real_y[:, 0] = 1

            acc_real = calc_acc(d_real_pred, 'real')
            d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
            d_loss_real = torch.div(d_loss_real, cfg.batch_size)
            d_loss = d_loss_real + d_loss_ref

            d_loss.backward()
            self.opt_D.step()
            
            self.opt_D_seg.zero_grad()
            real_seg_image_batch, syn_seg_image_batch = self.get_next_seg_batches()

            # ============ ref image D ====================================================
            ref_image_batch = self.R(syn_seg_image_batch)
            d_ref_pred = self.D_seg(ref_image_batch).view(-1,2)
            d_ref_y = torch.zeros(d_ref_pred.size(), dtype=torch.float, device=cfg.dev)
            d_ref_y[:, 1] = 1

            acc_ref = calc_acc(d_ref_pred, 'refine')
            d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
            d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)

            # ============ real image D ====================================================
            d_real_pred = self.D_seg(real_seg_image_batch).view(-1,2)
            d_real_y = torch.zeros(d_real_pred.size(), dtype=torch.float, device=cfg.dev)
            d_real_y[:, 0] = 1

            acc_real = calc_acc(d_real_pred, 'real')
            d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
            d_loss_real = torch.div(d_loss_real, cfg.batch_size)
            d_loss_seg = d_loss_real + d_loss_ref

            d_loss_seg.backward()
            self.opt_D.step()

    def train(self):
        logging.info('=' * 50)
        logging.info('Training...')
        image_history_buffer = ImageHistoryBuffer((0, cfg.img_channels, cfg.img_height, cfg.img_width),
                                                  cfg.buffer_size * 10, cfg.batch_size)
        self.val_synth_batch = self.get_next_synth_batch()[:32]
        self.val_real_batch = self.get_next_real_batch()[:32]

        wandb.watch(self.R)
        wandb.watch(self.D)

        self.print_training_setup()
        self.metrics_calculation_thread = threading.Thread(target=self.metrics_thread, args=())
        self.metrics_calculation_thread.start()

        for step in range(cfg.train_steps):
            logging.info('Step[%d/%d]' % (step, cfg.train_steps))

            # ========= train the R =========
            self.D.eval()
            self.D_seg.eval()
            self.R.train()

            total_r_loss = 0.0
            total_r_loss_reg_scale = 0.0
            total_r_loss_adv = 0.0
            total_r_loss_adv_seg = 0.0
            total_acc_adv = 0.0
            total_acc_adv_seg = 0.0
            
            for index in range(cfg.k_r):
                self.opt_R.zero_grad()

                syn_image_batch = self.get_next_synth_batch()

                ref_image_batch = self.R(syn_image_batch)

                d_ref_pred = self.D(ref_image_batch).view(-1, 2)
                d_ref_seg_pred = self.D_seg(ref_image_batch).view(-1, 2)
                d_real_y = torch.zeros(d_ref_pred.size(), dtype=torch.float, device=cfg.dev)
                d_real_y[:, 0] = 1.
                # d_ref_y = d_ref_y.softmax(dim=1)

                acc_adv = calc_acc(d_ref_pred, 'real')
                acc_adv_seg = calc_acc(d_ref_seg_pred, 'real')

                r_loss_reg = self.self_regularization_loss(ref_image_batch, syn_image_batch)
                r_loss_reg_scale = torch.mul(r_loss_reg, self.delta)
                r_loss_reg_scale = torch.div(r_loss_reg_scale, cfg.batch_size)

                r_loss_adv = self.local_adversarial_loss(d_ref_pred, d_real_y)
                r_loss_adv = torch.div(r_loss_adv, cfg.batch_size)

                r_loss_adv_seg = self.local_adversarial_loss(d_ref_seg_pred, d_real_y)
                r_loss_adv_seg = torch.mul(r_loss_adv_seg, 0.5)
                r_loss_adv_seg = torch.div(r_loss_adv_seg, cfg.batch_size)

                r_loss = r_loss_reg_scale + r_loss_adv + r_loss_adv_seg

                r_loss.backward()
                self.opt_R.step()

                total_r_loss += r_loss
                total_r_loss_reg_scale += r_loss_reg_scale
                total_r_loss_adv += r_loss_adv
                total_r_loss_adv_seg += r_loss_adv_seg
                total_acc_adv += acc_adv
                total_acc_adv_seg += acc_adv_seg

            mean_r_loss = total_r_loss / cfg.k_r
            mean_r_loss_reg_scale = total_r_loss_reg_scale / cfg.k_r
            mean_r_loss_adv = total_r_loss_adv / cfg.k_r
            mean_r_loss_adv_seg = total_r_loss_adv_seg / cfg.k_r
            mean_acc_adv = total_acc_adv / cfg.k_r
            mean_acc_adv_seg = total_acc_adv_seg / cfg.k_r

            # ========= train the D =========
            self.R.eval()
            self.D.train()
            self.D_seg.train()

            total_d_loss_real = 0.0
            total_d_loss_ref = 0.0
            total_d_loss = 0.0
            total_d_accuracy_real = 0.0
            total_d_accuracy_ref = 0.0

            total_d_loss_real_seg = 0.0
            total_d_loss_ref_seg = 0.0
            total_d_loss_seg = 0.0
            total_d_accuracy_real_seg = 0.0
            total_d_accuracy_ref_seg = 0.0


            for index in range(cfg.k_d):

                real_image_batch, syn_image_batch = self.get_next_batches()
                
                ref_image_batch = self.R(syn_image_batch)

                # use a history of refined images
                half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                image_history_buffer.add_to_image_history_buffer(ref_image_batch.cpu().data.numpy().transpose((0, 1, 3, 2)))

                if len(half_batch_from_image_history):
                    torch_type = torch.from_numpy(half_batch_from_image_history)
                    v_type = Variable(torch_type).cuda(cfg.cuda_num).transpose(2, 3)
                    ref_image_batch[:cfg.batch_size // 2] = v_type

                d_real_pred = self.D(real_image_batch).view(-1,2)
                d_real_y = torch.zeros(d_real_pred.size(), dtype=torch.float, device=cfg.dev)
                d_real_y[:, 0] = 1.
                d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
                d_loss_real = torch.div(d_loss_real, cfg.batch_size)

                acc_real = calc_acc(d_real_pred, 'real')

                d_ref_pred = self.D(ref_image_batch).view(-1, 2)
                d_ref_y = torch.zeros(d_real_pred.size(), dtype=torch.float, device=cfg.dev)
                d_ref_y[:, 1] = 1.
                d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
                d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)

                acc_ref = calc_acc(d_ref_pred, 'refine')

                d_loss = d_loss_real + d_loss_ref

                total_d_loss_real += d_loss_real.item()
                total_d_loss_ref += d_loss_ref.item()
                total_d_loss += d_loss.item()
                total_d_accuracy_real += acc_real
                total_d_accuracy_ref += acc_ref

                self.D.zero_grad()
                d_loss.backward()
                self.opt_D.step()

                real_seg_image_batch, syn_seg_image_batch = self.get_next_seg_batches()
                ref_image_batch = self.R(syn_seg_image_batch)
                # ref_image_batch[:cfg.batch_size // 2] = v_type

                d_real_pred = self.D_seg(real_seg_image_batch).view(-1,2)
                d_real_y = torch.zeros(d_real_pred.size(), dtype=torch.float, device=cfg.dev)
                d_real_y[:, 0] = 1.
                d_loss_real_seg = self.local_adversarial_loss(d_real_pred, d_real_y)
                d_loss_real_seg = torch.div(d_loss_real_seg, cfg.batch_size)

                acc_real_seg = calc_acc(d_real_pred, 'real')

                d_ref_pred = self.D_seg(ref_image_batch).view(-1, 2)
                d_ref_y = torch.zeros(d_real_pred.size(), dtype=torch.float, device=cfg.dev)
                d_ref_y[:, 1] = 1.
                d_loss_ref_seg = self.local_adversarial_loss(d_ref_pred, d_ref_y)
                d_loss_ref_seg = torch.div(d_loss_ref_seg, cfg.batch_size)

                acc_ref_seg = calc_acc(d_ref_pred, 'refine')

                d_loss_seg = d_loss_real_seg + d_loss_ref_seg

                total_d_loss_real_seg += d_loss_real_seg.item()
                total_d_loss_ref_seg += d_loss_ref_seg.item()
                total_d_loss_seg += d_loss_seg.item()
                total_d_accuracy_real_seg += acc_real_seg
                total_d_accuracy_ref_seg += acc_ref_seg

                self.D_seg.zero_grad()
                d_loss_seg.backward()
                self.opt_D_seg.step()

            mean_d_loss_real = total_d_loss_real / cfg.k_d
            mean_d_loss_ref = total_d_loss_ref / cfg.k_d
            mean_d_loss = total_d_loss / cfg.k_d
            mean_d_accuracy_real = total_d_accuracy_real / cfg.k_d
            mean_d_accuracy_ref = total_d_accuracy_ref / cfg.k_d
            
            mean_d_loss_real_seg = total_d_loss_real_seg / cfg.k_d
            mean_d_loss_ref_seg = total_d_loss_ref_seg / cfg.k_d
            mean_d_loss_seg = total_d_loss_seg / cfg.k_d
            mean_d_accuracy_real_seg = total_d_accuracy_real_seg / cfg.k_d
            mean_d_accuracy_ref_seg = total_d_accuracy_ref_seg / cfg.k_d

            l1_dict = {
                "training_step": step,
                "refiner" : {
                    "loss": {
                        "adv": mean_r_loss_adv.item(),
                        "adv_seg": mean_r_loss_adv_seg.item(),
                        "l1reg": mean_r_loss_reg_scale.item(),
                        "total": mean_r_loss.item(),
                    },
                    "accuracy": mean_acc_adv.item(),
                    "accuracy_seg": mean_acc_adv_seg.item()
                },
                "disciminator": {
                    "normal" : {
                        "loss": {
                            "real": mean_d_loss_real,
                            "ref": mean_d_loss_ref,
                            "total": mean_d_loss,
                        },
                        "accuracy": {
                            "real": mean_d_accuracy_real,
                            "ref": mean_d_accuracy_ref,
                        },
                    },
                    "seg": {
                        "loss": {
                            "real": mean_d_loss_real_seg,
                            "ref": mean_d_loss_ref_seg,
                            "total": mean_d_loss_seg,
                        },
                        "accuracy": {
                            "real": mean_d_accuracy_real_seg,
                            "ref": mean_d_accuracy_ref_seg,
                        },
                    },
                },
            }

            self.R.eval()
            self.D.eval()
            self.D_seg.eval()
            val_ref_batch = self.R(self.val_synth_batch).clone()

            data = {'step': step,
                    'l1_dict': copy.deepcopy(l1_dict),
                    'val_ref_batch': val_ref_batch.detach().clone()}

            self.metrics_queue.put(copy.deepcopy(data))
            # if step % cfg.save_per == 0:
            #     logging.info('Save two model dict.')
            #     torch.save(self.D.state_dict(), f'{cfg.train_res_path}/{cfg.D_path}' % step)
            #     torch.save(self.R.state_dict(), f'{cfg.train_res_path}/{cfg.R_path}' % step)

            # if step % cfg.save_per == 0:
            #     self.R.eval()
            #     self.D.eval()

            #     val_ref_image_batch = self.R(self.val_synth_batch)

            #     logging.info('Save two model dict.')
            #     torch.save(self.D.state_dict(), f'{cfg.train_res_path}/{cfg.D_path}' % step)
            #     torch.save(self.R.state_dict(), f'{cfg.train_res_path}/{cfg.R_path}' % step)

            #     self.generate_batch_train_image(self.val_synth_batch, val_ref_image_batch, self.val_real_batch, step_index=step)

            #     kl_score = calculate_kl_score(self.val_synth_batch.cpu().data,
            #                                 val_ref_image_batch.cpu().data,
            #                                 self.val_real_batch.cpu().data)
            #     if cfg.fretchet:
            #         # self.inception_model.to(cfg.dev)
            #         fretchet_dist_real_ref = calculate_fretchet(real_image_batch,
            #                                                 val_ref_image_batch,
            #                                                 self.inception_model)
            #         fretchet_dist_synth_ref = calculate_fretchet(syn_image_batch,
            #                                                 val_ref_image_batch,
            #                                                 self.inception_model)
            #         fretchet_score = {
            #             "fretcher" :{
            #                 'real-ref': fretchet_dist_real_ref,
            #                 'synth-ref': fretchet_dist_synth_ref,
            #             }
            #         }
            #         # self.inception_model.to('cpu')
            #     else:
            #         fretchet_score = {}
            #     log_dict = kl_score | l1_dict | fretchet_score
            # else:
            #     log_dict = l1_dict

            # wandb.log(log_dict)

    def  generate_batch_train_image(self, syn_image_batch, ref_image_batch, real_image_batch, step_index=-1):
        logging.info('=' * 50)
        logging.info('Generating a batch of training images...')

        pic_path = os.path.join(cfg.train_res_path, 'step_%d.png' % step_index)
        hist_path = os.path.join(cfg.train_res_path, 'step_%d_hist.png' % step_index)
        img = generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, pic_path)
        hist, kl_scores = generate_avg_histograms(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, hist_path)
        # generate_img_batch

        # img.save(pic_path, 'png')
        # hist.savefig(hist_path)

        wandb.log({f"preview": wandb.Image(img), f"preview_hist": wandb.Image(hist)})
        logging.info('=' * 50)
        plt.close(hist)

    # def generate_all_train_image(self):
    #     print('=' * 50)
    #     print('Generating all training images...')
    #     self.R.eval()
    #
    #     for index, (syn_image_batch, _) in enumerate(self.syn_train_loader):
    #         pic_path = os.path.join(cfg.final_res_path, 'batch_%d.png' % index)
    #
    #         syn_image_batch = Variable(syn_image_batch, volatile=True).cuda(cfg.cuda_num)
    #         ref_image_batch = self.R(syn_image_batch)
    #         generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, pic_path)
    #     print('=' * 50)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(42)

    signal.signal(signal.SIGTERM, terminate_signal)
    obj = Main()
    obj.build_network()

    obj.load_data()

    obj.pre_train_r()
    obj.pre_train_d()

    obj.train()
    wandb.finish()
    # obj.generate_all_train_image()
