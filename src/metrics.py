import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np


def compute_metrics(hrs, srs):
    # convert the two batches from rgb to YCrCb
    hrs = np.asarray([cv2.cvtColor(hr, cv2.COLOR_RGB2YCR_CB) for hr in hrs])
    srs = np.asarray([cv2.cvtColor(sr, cv2.COLOR_RGB2YCR_CB) for sr in srs])

    # get only the Y channels of the batches
    hrs = hrs[:, :, :, 0]
    srs = srs[:, :, :, 0]

    # compute the psrn and ssim scores for all the samples
    n = hrs.shape[0]
    psnr = np.asarray([peak_signal_noise_ratio(hrs[i], srs[i]) for i in range(n)])
    ssim = np.asarray([structural_similarity(hrs[i], srs[i]) for i in range(n)])

    return psnr, ssim
