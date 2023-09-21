import torch
import torch.utils.data as Data
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from lib.image_history_buffer import ImageHistoryBuffer
from lib.network import Discriminator, Refiner
from lib.image_utils import generate_img_batch, generate_avg_histograms, calc_acc
import config as cfg
import os
import matplotlib.pyplot as plt
import logging
import wandb
import numpy as np
import h5py
from PIL import Image
import signal, sys

def terminate_signal(signalnum, handler):
    print ('Terminate the process')
    # save results, whatever...
    wandb.finish()
    sys.exit()

class HDF5Dataset(Data.Dataset):
    

    def __init__(self, file_path, datasets: tuple, transform=None):
        super().__init__()

        self.file_path = file_path
        self.datasets = datasets

        self.transform = transform

        with h5py.File(self.file_path) as file:
            self.length = len(file[self.datasets[0]])
            self.target = file[self.datasets[1]][0]
            # print(f'self.target: {self.target}')

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index) :
        if not hasattr(self, '_hf'):
            self._open_hdf5()
        # print(index)
        # with h5py.File(self.file_path, 'r') as file:

        #     x = file[self.datasets[0]][index]
        #     x = Image.fromarray(np.array(x))

        #     if self.transform:
        #         x = self.transform(x)
        #     else:
        #         x = torch.from_numpy(x)

        #     y = file[self.datasets[1]][index]
        #     target = torch.zeros(1,2)
        #     target = target[:, y] = 1

        # return (x, y)
        # get data
        x = self._hf[self.datasets[0]][index]
        x = Image.fromarray(np.array(x))
        # x = self.data[index]
        # x = Image.fromarray(np.array(x))
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # y = self._hf[self.datasets[1]][index]
        # y = self.target
        # target = torch.zeros(1,2)
        # y = target[:, y] = 1
        # return (x, y)
        return x

    def __len__(self):
        return self.length
    

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
            }
)


    def get_next_synth_batch(self):
        try:
            syn_image_batch  = next(self.syn_train_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader 
            self.syn_train_iter = iter(self.syn_train_loader)
            syn_image_batch = next(self.syn_train_iter)

        syn_image_batch = syn_image_batch.to(cfg.dev)
        return syn_image_batch

    def get_next_real_batch(self):
        try:
            real_image_batch = next(self.real_image_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader 
            self.real_image_iter = iter(self.real_loader)
            real_image_batch  = next(self.real_image_iter)

        real_image_batch = real_image_batch.to(cfg.dev)
        return real_image_batch
    
    def reset_iters(self):
        self.real_image_iter = iter(self.real_loader)
        self.syn_train_iter = iter(self.syn_train_loader)

    def get_next_batches(self):
        real_batch = self.get_next_real_batch()
        synth_batch = self.get_next_synth_batch()

        if real_batch.size(0) != synth_batch.size(0):
            self.reset_iters()
            real_batch = self.get_next_real_batch()
            synth_batch = self.get_next_synth_batch()

        return (real_batch, synth_batch)
    
    def build_network(self):
        logging.info('=' * 50)
        logging.info('Building network...')
        self.R = Refiner(4, cfg.img_channels, nb_features=64)
        self.D = Discriminator(input_features=cfg.img_channels)

        if cfg.cuda_use:
            self.R.cuda(cfg.cuda_num)
            self.D.cuda(cfg.cuda_num)

        self.opt_R = torch.optim.SGD(self.R.parameters(), lr=cfg.r_lr)
        self.opt_D = torch.optim.SGD(self.D.parameters(), lr=cfg.d_lr)
        # self.opt_R = torch.optim.Adam(self.R.parameters(), lr=cfg.r_lr)
        # self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.d_lr)
        # self.self_regularization_loss = nn.L1Loss(size_average=False)
        # self.local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)
        self.self_regularization_loss = nn.L1Loss(reduction='sum')
        self.local_adversarial_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.delta = cfg.delta
        self.r_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_R, step_size=cfg.r_step_size, gamma=cfg.r_gamma)
        self.d_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_D, step_size=cfg.d_step_size, gamma=cfg.d_gamma)

        self.val_real_batch = None
        self.val_synth_batch = None

        self.d_output_size = self.D(torch.rand((cfg.batch_size, cfg.img_channels, cfg.img_height, cfg.img_width), device=cfg.dev)).size()

    def load_data(self):
        logging.info('=' * 50)
        logging.info('Loading data...')
        self.transform = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize((cfg.img_width, cfg.img_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])

        # syn_train_folder = torchvision.datasets.ImageFolder(root=cfg.syn_path, transform=transform)
        syn_train_folder = HDF5Dataset(cfg.syn_path, cfg.syn_datasets, self.transform)
        # print(syn_train_folder)
        self.syn_train_loader = Data.DataLoader(syn_train_folder, batch_size=cfg.batch_size, shuffle=True,
                                                pin_memory=True, num_workers=4)
        self.syn_train_iter = iter(self.syn_train_loader)
        logging.info('syn_train_batch %d' % len(self.syn_train_loader))

        # real_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transform)
        real_folder = HDF5Dataset(cfg.real_path,, cfg.real_datesets, self.transform)
        # real_folder.imgs = real_folder.imgs[:2000]
        self.real_loader = Data.DataLoader(real_folder, batch_size=cfg.batch_size, shuffle=True,
                                           pin_memory=True, num_workers=4)
        self.real_image_iter = iter(self.real_loader)
        logging.info('real_batch %d' % len(self.real_loader))

    def pre_train_r(self):
        logging.info('=' * 50)
        if cfg.ref_pre_path:
            logging.info('Loading R_pre from %s' % cfg.ref_pre_path)
            self.R.load_state_dict(torch.load(cfg.ref_pre_path))
            return

        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        logging.info('pre-training the refiner network %d times...' % cfg.r_pretrain)

        for index in range(cfg.r_pretrain):

            syn_image_batch = self.get_next_synth_batch()

            self.R.train()
            ref_image_batch = self.R(syn_image_batch)

            r_loss = self.self_regularization_loss(ref_image_batch, syn_image_batch)
            # r_loss = torch.div(r_loss, cfg.batch_size)
            r_loss = torch.mul(r_loss, self.delta)

            self.opt_R.zero_grad()
            r_loss.backward()
            self.opt_R.step()

            # log every `log_interval` steps
            if (index % cfg.r_pre_per == 0) or (index == cfg.r_pretrain - 1):
                # figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(index)
                logging.info('[%d/%d] (R)reg_loss: %.4f' % (index, cfg.r_pretrain, r_loss.item()))
                
                syn_image_batch = self.get_next_synth_batch()
                real_image_batch = self.get_next_real_batch()

                self.R.eval()
                ref_image_batch = self.R(syn_image_batch)

                figure_path = os.path.join(cfg.train_res_path, 'refined_image_batch_pre_train_%d.png' % index)
                generate_img_batch(syn_image_batch.data.cpu(), ref_image_batch.data.cpu(),
                                   real_image_batch.data.cpu(), figure_path)
                self.R.train()

                logging.info('Save R_pre to models/R_pre.pkl')
                torch.save(self.R.state_dict(), f'{cfg.train_res_path}/models/R_pre.pkl')

    def pre_train_d(self):
        logging.info('=' * 50)
        if cfg.disc_pre_path:
            logging.info('Loading D_pre from %s' % cfg.disc_pre_path)
            self.D.load_state_dict(torch.load(cfg.disc_pre_path))
            return

        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        logging.info('pre-training the discriminator network %d times...' % cfg.r_pretrain)
        
        self.D.train()
        self.R.eval()

        for index in range(cfg.d_pretrain):
    
            real_image_batch, syn_image_batch = self.get_next_batches()

            # ============ real image D ====================================================
            self.D.train()
            d_real_pred = self.D(real_image_batch).view(-1,2)
            d_real_y = torch.zeros(d_real_pred.size(), dtype=torch.float, device=cfg.dev)
            d_real_y[:, 0] = 1

            acc_real = calc_acc(d_real_pred, 'real')
            
            d_loss_real = self.local_adversarial_loss(d_real_pred, d_real_y)
            # d_loss_real = torch.div(d_loss_real, cfg.batch_size)

            # ============ syn image D ====================================================
            self.R.eval()
            ref_image_batch = self.R(syn_image_batch)
            
            self.D.train()
            d_ref_pred = self.D(ref_image_batch).view(-1,2)
            d_ref_y = torch.zeros(d_ref_pred.size(), dtype=torch.float, device=cfg.dev)
            d_ref_y[:, 1] = 1

            acc_ref = calc_acc(d_ref_pred, 'refine')
            d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
            d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)
            
            d_loss = d_loss_real + d_loss_ref

            self.opt_D.zero_grad()
            d_loss.backward()
            self.opt_D.step()

            if (index % cfg.d_pre_per == 0) or (index == cfg.d_pretrain - 1):
                logging.info('[%d/%d] (D)d_loss:%f  acc_real:%.2f%% acc_ref:%.2f%%'
                      % (index, cfg.d_pretrain, d_loss.item(), acc_real, acc_ref))
            
        logging.info('Save D_pre to models/D_pre.pkl')
        torch.save(self.D.state_dict(), f'{cfg.train_res_path}/models/D_pre.pkl')


    def train(self):
        logging.info('=' * 50)
        logging.info('Training...')
        image_history_buffer = ImageHistoryBuffer((0, cfg.img_channels, cfg.img_height, cfg.img_width),
                                                  cfg.buffer_size * 10, cfg.batch_size)
        
        self.val_synth_batch = self.get_next_synth_batch()
        self.val_real_batch = self.get_next_real_batch()

        wandb.watch(self.R)
        wandb.watch(self.D)

        for step in range(cfg.train_steps):
            logging.info('Step[%d/%d]' % (step, cfg.train_steps))

            # ========= train the R =========
            self.D.eval()
            self.R.train()

            total_r_loss = 0.0
            total_r_loss_reg_scale = 0.0
            total_r_loss_adv = 0.0
            total_acc_adv = 0.0

            for index in range(cfg.k_r):
                # self.opt_R.zero_grad()

                syn_image_batch = self.get_next_synth_batch()

                ref_image_batch = self.R(syn_image_batch)

                d_ref_pred = self.D(ref_image_batch).view(-1, 2)
                d_real_y = torch.zeros(d_ref_pred.size(), dtype=torch.float, device=cfg.dev)
                d_real_y[:, 0] = 1.
                # d_ref_y = d_ref_y.softmax(dim=1)

                acc_adv = calc_acc(d_ref_pred, 'real')

                r_loss_reg = self.self_regularization_loss(ref_image_batch, syn_image_batch)
                r_loss_reg_scale = torch.mul(r_loss_reg, self.delta)
                # r_loss_reg_scale = torch.div(r_loss_reg_scale, cfg.batch_size)

                r_loss_adv = self.local_adversarial_loss(d_ref_pred, d_real_y)
                # r_loss_adv = torch.div(r_loss_adv, cfg.batch_size)

                r_loss = r_loss_reg_scale + r_loss_adv

                self.opt_R.zero_grad()
                r_loss.backward()
                self.opt_R.step()

                total_r_loss += r_loss / cfg.batch_size
                total_r_loss_reg_scale += r_loss_reg_scale / cfg.batch_size
                total_r_loss_adv += r_loss_adv / cfg.batch_size
                total_acc_adv += acc_adv

            mean_r_loss = total_r_loss / cfg.k_r
            mean_r_loss_reg_scale = total_r_loss_reg_scale / cfg.k_r
            mean_r_loss_adv = total_r_loss_adv / cfg.k_r
            mean_acc_adv = total_acc_adv / cfg.k_r

            # logging.info(f'(R)r_loss: {mean_r_loss.item():.4f}, \
            #                 r_loss_reg: {mean_r_loss_reg_scale.item():.4f}, \
            #                 r_loss_adv: {mean_r_loss_adv.item():.4f}({mean_acc_adv:.2f})')
            
            # ========= train the D =========
            self.R.eval()
            self.D.train()

            total_d_loss_real = 0.0
            total_d_loss_ref = 0.0
            total_d_loss = 0.0
            total_d_accuracy_real = 0.0
            total_d_accuracy_ref = 0.0

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
                # d_loss_real = torch.div(d_loss_real, cfg.batch_size)

                acc_real = calc_acc(d_real_pred, 'real')

                d_ref_pred = self.D(ref_image_batch).view(-1, 2)
                d_ref_y = torch.ones(d_real_pred.size(), dtype=torch.float, device=cfg.dev)
                d_ref_y[:, 1] = 1.
                d_loss_ref = self.local_adversarial_loss(d_ref_pred, d_ref_y)
                # d_loss_ref = torch.div(d_loss_ref, cfg.batch_size)
                acc_ref = calc_acc(d_ref_pred, 'refine')

                d_loss = d_loss_real + d_loss_ref

                total_d_loss_real += d_loss_real.item() / cfg.batch_size
                total_d_loss_ref += d_loss_ref.item() / cfg.batch_size
                total_d_loss += d_loss.item() / cfg.batch_size
                total_d_accuracy_real += acc_real
                total_d_accuracy_ref += acc_ref

                self.D.zero_grad()
                d_loss.backward()
                self.opt_D.step()

                # logging.info(f'(D)d_loss: {total_d_loss.item():.4f}, \
                #                 d_loss_real: {mean_r_loss_reg_scale.item():.4f}, \
                #                 d_loss_refine: {mean_r_loss_adv.item():.4f}({mean_acc_adv:.2f})')

            mean_d_loss_real = total_d_loss_real / cfg.k_d
            mean_d_loss_ref = total_d_loss_ref / cfg.k_d
            mean_d_loss = total_d_loss / cfg.k_d
            mean_d_accuracy_real = total_d_accuracy_real / cfg.k_d
            mean_d_accuracy_ref = total_d_accuracy_ref / cfg.k_d

            log_dict = {
                "training_step": step,
                "refiner" : {
                    "loss": {
                        "adv": mean_r_loss_adv.item(),
                        "l1reg": mean_r_loss_reg_scale.item(),
                        "total": mean_r_loss.item(),
                    },
                    "accuracy": mean_acc_adv.item(),
                },
                "disciminator": {
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
            }

            wandb.log(log_dict)

            if step % cfg.save_per == 0:
                logging.info('Save two model dict.')
                torch.save(self.D.state_dict(), f'{cfg.train_res_path}/{cfg.D_path}' % step)
                torch.save(self.R.state_dict(), f'{cfg.train_res_path}/{cfg.R_path}' % step)

                self.R.eval()
                ref_image_batch = self.R(self.val_synth_batch)
                self.generate_batch_train_image(self.val_synth_batch, ref_image_batch, self.val_real_batch, step_index=step)


    def  generate_batch_train_image(self, syn_image_batch, ref_image_batch, real_image_batch, step_index=-1):
        logging.info('=' * 50)
        logging.info('Generating a batch of training images...')
        self.R.eval()

        pic_path = os.path.join(cfg.train_res_path, 'step_%d.png' % step_index)
        hist_path = os.path.join(cfg.train_res_path, 'step_%d_hist.png' % step_index)
        img = generate_img_batch(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, pic_path)
        hist, kl_score = generate_avg_histograms(syn_image_batch.cpu().data, ref_image_batch.cpu().data, real_image_batch.cpu().data, hist_path)
        # generate_img_batch
        img.save(pic_path, 'png')
        hist.savefig(hist_path)
        wandb.log(kl_score)
        wandb.log({f"prewiev": wandb.Image(img), f"prewiev_hist": wandb.Image(hist)})
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

    signal.signal(signal.SIGTERM, terminate_signal)
    obj = Main()
    obj.build_network()

    obj.load_data()

    obj.pre_train_r()
    obj.pre_train_d()

    obj.train()
    wandb.finish()
    # obj.generate_all_train_image()
