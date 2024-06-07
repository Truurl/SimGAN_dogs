import os
import torch

fretchet = True
attention = False

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
metric_dev = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
# Is the PC has cuda
cuda_use = torch.cuda.is_available()
# which cuda to use
cuda_num = 0


# learning rate for D, the lr in Apple blog is 0.0001
d_lr = 0.0001
# learning rate for R, the lr in Apple blog is 0.0001
r_lr = 0.0001
# lambda in paper, the author of the paper said it's 0.01
delta = 0.01

r_step_size = 1000
r_gamma = 0.7

d_step_size = 1000
d_gamma = 0.5

img_width = 64
img_height = 64
img_channels = 3

n_resnets = 4
nb_features = 64

n_heads = 4

# result show in 4 sample per line
pics_line = 4

# =================== training params ======================
# pre-train R times
r_pretrain = 1000
# pre-train D times
d_pretrain = 1000
# train steps
train_steps = 100000

batch_size = 256
# the history buffer size
# buffer_size = 12800
buffer_size = batch_size * 10
k_d = 1  # number of discriminator updates per step
k_r = 50  # number of generative network updates per step, the author of the paper said it's 50

# output R pre-training result per times
r_pre_per = 10
# output D pre-training result per times
d_pre_per = 10
# save model dictionary and training dataset output result per train times
save_per = 20


# pre-training dictionary path
# ref_pre_path = 'models/R_pre.pkl'
ref_pre_path = None
# disc_pre_path = 'models/D_pre.pkl'
disc_pre_path = None

# dictionary saving path
D_path = 'models/D_%d.pkl'
R_path = 'models/R_%d.pkl'

# synthetic image path
# syn_path = f"{os.getenv('MEMFS')}/dataset/UnityEyes.hdf5"
# syn_path = f"{os.getenv('MEMFS')}/dataset/UnityEyes.dat"
# syn_path = f"{os.getenv('MEMFS')}/dataset/SynthEyes.dat"
syn_path = f"{os.getenv('MEMFS')}/dataset/synth_dogs.hdf5"
syn_datasets = ('synth_img', 'symth_labels')
# real image path
# real_path =f"{os.getenv('MEMFS')}/dataset/MPIIGaze.dat"
real_path =f"{os.getenv('MEMFS')}/dataset/real_dogs_green.hdf5"
real_datasets = ('real_img', 'real_labels')

# training result path to save
train_res_path = f"{os.getenv('SCRATCH')}/simgan_results/sgd_batch_{batch_size}_delta_{delta}_D_{d_lr}_{k_d}__R_{r_lr}_{k_r}"
# final_res_path = 'final_res'
