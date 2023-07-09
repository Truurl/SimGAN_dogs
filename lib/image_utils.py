import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import config as cfg
import wandb
import matplotlib.pyplot as plt

def normalize_img(img):
    # img_type: numpy
    img = img * 1.0 / 255
    return (img - 0.5) / 0.5


def restore_img(img):
    # img_type: numpy
    img += max(-img.min(), 0)
    if img.max() != 0:
        img /= img.max()
    img *= 255
    img = img.astype(np.uint8)
    return img


def generate_avg_histograms(syn_batch, ref_batch, real_batch, hist_path):
    
    def tensor_to_numpy(img):
        img = img.numpy()
        img += max(-img.min(), 0)
        if img.max() != 0:
            img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        img = np.transpose(img, [1, 2, 0])
        return img

    num_images = min(syn_batch.size()[0], ref_batch.size()[0], real_batch.size()[0])

    synth_histograms = []
    ref_histograms = []
    real_histograms = []

    for index in range(num_images):
        synth_image = tensor_to_numpy(syn_batch[index, :, :, :])
        ref_image = tensor_to_numpy(ref_batch[index, :, :, :])
        real_image = tensor_to_numpy(real_batch[index, :, :, :])

        synth_hist, _ = np.histogram(synth_image.ravel(), bins=256, range=(0, 255))
        ref_hist, _ = np.histogram(ref_image.ravel(), bins=256, range=(0, 256))
        real_hist, _ = np.histogram(real_image.ravel(), bins=256, range=(0, 255))

        synth_histograms.append(synth_hist)
        ref_histograms.append(ref_hist)
        real_histograms.append(real_hist)

    avg_synth_hist = np.mean(synth_histograms, axis=0)
    avg_ref_hist = np.mean(ref_histograms, axis=0)
    avg_real_hist = np.mean(real_histograms, axis=0)
    avg_real_hist[0:10] = 0

    relative_entropy = avg_synth_hist * np.log(avg_synth_hist / avg_ref_hist, where=avg_synth_hist != 0)
    # kl_synth_ref = np.sum(avg_synth_hist * np.log(avg_synth_hist / avg_ref_hist))
    kl_synth_ref = np.sum(relative_entropy, where=np.isfinite(relative_entropy))

    relative_entropy = avg_synth_hist * np.log(avg_synth_hist / avg_real_hist, where=avg_synth_hist != 0)
    # kl_synth_real = np.sum(avg_synth_hist * np.log(avg_synth_hist / avg_real_hist))
    kl_synth_real =  np.sum(relative_entropy, where=np.isfinite(relative_entropy))

    relative_entropy = avg_ref_hist * np.log(avg_ref_hist / avg_real_hist, where=avg_ref_hist != 0)
    # kl_ref_real = np.sum(avg_ref_hist * np.log(avg_ref_hist / avg_real_hist))
    kl_ref_real =  np.sum(relative_entropy, where=np.isfinite(relative_entropy))

    
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle(f'kl_synth_ref: {kl_synth_ref:.4}, kl_synth_real: {kl_synth_real:.4}, kl_ref_real: {kl_ref_real:.4}')
 
    axs[0].plot(avg_synth_hist)
    axs[0].set_title('synth_avg_hist')
    
    axs[1].plot(avg_ref_hist)
    axs[1].set_title('ref_avg_hist')
    
    axs[2].plot(avg_real_hist)
    axs[2].set_title('real_avg_hist')

    return fig, {'kl_synth_ref': kl_synth_ref, 'kl_synth_real': kl_synth_real, 'kl_ref_real': kl_ref_real}
    # plt.savefig(hist_path)

def generate_img_batch(syn_batch, ref_batch, real_batch, png_path):
    # syn_batch_type: Tensor, ref_batch_type: Tensor
    def tensor_to_numpy(img):
        img = img.numpy()
        img += max(-img.min(), 0)
        if img.max() != 0:
            img /= img.max()
        img *= 255
        img = img.astype(np.uint8)
        img = np.transpose(img, [1, 2, 0])
        return img

    syn_batch = syn_batch[:64]
    ref_batch = ref_batch[:64]
    real_batch = real_batch[:64]

    a_blank = torch.zeros(cfg.img_height, cfg.img_width*2, 1).numpy().astype(np.uint8)

    nb = syn_batch.size(0)
    # print(syn_batch.size())
    # print(ref_batch.size())
    vertical_list = []

    for index in range(0, nb, cfg.pics_line):
        st = index
        end = st + cfg.pics_line

        if end > nb:
            end = nb

        syn_line = syn_batch[st:end]
        ref_line = ref_batch[st:end]
        diff_line = ref_batch[st:end] - syn_batch[st:end]
        real_line = real_batch[st:end]
        # print('====>', nb)
        # print(syn_line.size())
        # print(ref_line.size())
        nb_per_line = syn_line.size(0)

        line_list = []

        for i in range(nb_per_line):
            #print(i, len(syn_line))
            syn_np = tensor_to_numpy(syn_line[i])
            ref_np = tensor_to_numpy(ref_line[i])
            diff_np = tensor_to_numpy(diff_line[i])
            real_np = tensor_to_numpy(real_line[i])
            a_group = np.concatenate([syn_np, ref_np, diff_np, real_np], axis=1)
            line_list.append(a_group)


        fill_nb = cfg.pics_line - nb_per_line
        while fill_nb:
            line_list.append(a_blank)
            fill_nb -= 1
        # print(len(line_list))
        # print(line_list[0].shape)
        # print(line_list[1].shape)
        # print(line_list[2].shape)
        # print(line_list[3].shape)
        # print(line_list[4].shape)
        line = np.concatenate(line_list, axis=1)
        # print(line.dtype)
        vertical_list.append(line)

    imgs = np.concatenate(vertical_list, axis=0)
    if imgs.shape[-1] == 1:
        imgs = np.tile(imgs, [1, 1, 3])
    # print(imgs.shape, imgs.dtype)
    img = Image.fromarray(imgs)
    return img

def calc_acc(output, type='real'):
    assert type in ['real', 'refine']

    if type == 'real':
        label = torch.zeros(output.size(0), dtype=torch.long, device=cfg.dev)
    else:
        label = torch.ones(output.size(0), dtype=torch.long, device=cfg.dev)

    softmax_output = torch.nn.functional.softmax(output)
    acc = softmax_output.data.max(1)[1].cpu().numpy() == label.data.cpu().numpy()
    return acc.mean()
