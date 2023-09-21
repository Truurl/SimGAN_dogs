import os
import torch

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Is the PC has cuda
cuda_use = torch.cuda.is_available()
# which cuda to use
cuda_num = 0

# learning rate for D, the lr in Apple blog is 0.0001
d_lr = 0.0001
# learning rate for R, the lr in Apple blog is 0.0001
r_lr = 0.0001
# lambda in paper, the author of the paper said it's 0.01
delta = 0.1

gram_loss = False
kl_loss = True

delta_kl = 0.000001
delta_gram = 1000000000

r_step_size = 1000
r_gamma = 0.7

d_step_size = 1000
d_gamma = 0.5

img_width = 128
img_height = 128
img_channels = 3

# synthetic image path
syn_path = f"{os.getenv('MEMFS')}/dataset/synth_dogs.hdf5"
syn_datasets = ('synth_img', 'symth_labels')
# real image path
real_path = f"{os.getenv('MEMFS')}/dataset/real_dogs.hdf5"
real_datesets = ('real_img', 'real_labels')

# synthetic image path for segmented images
# syn_path = f"{os.getenv('MEMFS')}/dataset/synth_seg_dogs.hdf5"
# syn_datasets = ('synth_img', 'synth_labels')
# real image path for segmented images
# real_path = f"{os.getenv('MEMFS')}/dataset/real_seg_dogs.hdf5"
# real_datesets = ('real_img', 'real_labels')

# training result path to save
train_res_path = f"{os.getenv('SCRATCH')}/simgan_dogs/results/D_{d_lr}_{d_step_size}_{d_gamma}__R_{r_lr}_{r_step_size}_{r_gamma}"
# final_res_path = 'final_res'

# result show in 4 sample per line
pics_line = 2

# =================== training params ======================
# pre-train R times
r_pretrain = 1000
# pre-train D times
d_pretrain = 200
# train steps
train_steps = 100000

batch_size = 300
# test_batch_size = 128
# the history buffer size
buffer_size = batch_size * 10
k_d = 1  # number of discriminator updates per step
k_r = 50  # number of generative network updates per step, the author of the paper said it's 50

# output R pre-training result per times
r_pre_per = 50
# output D pre-training result per times
d_pre_per = 50
# save model dictionary and training dataset output result per train times
save_per = 10


# pre-training dictionary path
# ref_pre_path = 'models/R_pre.pkl'
ref_pre_path = None
# disc_pre_path = 'models/D_pre.pkl'
disc_pre_path = None

# dictionary saving path
D_path = 'models/D_%d.pkl'
R_path = 'models/R_%d.pkl'

same_pictures_path = '/net/people/plgrid/plgtrurl/SimGAN/SimGAN_pytorch_dogs/calibaration/same'
different_pictures_path = '/net/people/plgrid/plgtrurl/SimGAN/SimGAN_pytorch_dogs/calibaration/different'
synth_pictures_path = '/net/people/plgrid/plgtrurl/SimGAN/SimGAN_pytorch_dogs/calibaration/synth'
