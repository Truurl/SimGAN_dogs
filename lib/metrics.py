import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg

def calculate_activation_statistics(images, model, batch_size=128, dims=2048):
    model.eval()
    act=np.empty((len(images), dims))

    pred = model(images)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2


    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))


    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model)
     mu_2,std_2=calculate_activation_statistics(images_fake,model)

     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value

def calculate_kl_score(syn_batch, ref_batch, real_batch):

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
    num_channels = syn_batch.size(1)
    num_bins = (num_channels * 256)

    synth_histograms = np.zeros((num_bins,), dtype=float)
    ref_histograms = np.zeros((num_bins,), dtype=float)
    real_histograms = np.zeros((num_bins,), dtype=float)

    for index in range(num_images):
        synth_image = tensor_to_numpy(syn_batch[index, :, :, :])
        ref_image = tensor_to_numpy(ref_batch[index, :, :, :])
        real_image = tensor_to_numpy(real_batch[index, :, :, :])

        # offset channels for one continous histogram
        for chan in range(1, num_channels):
            synth_image[:, :, chan] += chan * 256
            ref_image[:, :, chan] += chan * 256
            real_image[:, :, chan] += chan * 256

        synth_hist, _ = np.histogram(synth_image.ravel(), bins=num_bins, range=(0, 256))
        ref_hist, _ = np.histogram(ref_image.ravel(), bins=num_bins, range=(0, 256))
        real_hist, _ = np.histogram(real_image.ravel(), bins=num_bins, range=(0, 256))

        synth_histograms += synth_hist
        ref_histograms += ref_hist
        real_histograms += real_hist

    avg_synth_hist = synth_histograms / num_images
    avg_ref_hist = ref_histograms / num_images
    avg_real_hist = real_histograms / num_images

    relative_entropy = avg_synth_hist * np.log(avg_synth_hist / avg_ref_hist, where=avg_synth_hist != 0)
    kl_synth_ref = np.sum(relative_entropy, where=np.isfinite(relative_entropy))

    relative_entropy = avg_synth_hist * np.log(avg_synth_hist / avg_real_hist, where=avg_synth_hist != 0)
    kl_synth_real =  np.sum(relative_entropy, where=np.isfinite(relative_entropy))

    relative_entropy = avg_ref_hist * np.log(avg_ref_hist / avg_real_hist, where=avg_ref_hist != 0)
    kl_ref_real =  np.sum(relative_entropy, where=np.isfinite(relative_entropy))

    return {
        'kl divergence': {
            'synth-ref': kl_synth_ref,
            'synth-real': kl_synth_real,
            'ref-real': kl_ref_real,
            },
        }
