import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import math


def compute_metrics(hrs, srs):
    # clip to 0-1 ranges
    srs = np.clip(srs, a_max=1, a_min=0)
    hrs = np.clip(hrs, a_max=1, a_min=0)

    # convert from rgb to YCrCb
    hrs = np.asarray([cv2.cvtColor(hr, cv2.COLOR_RGB2YCR_CB) for hr in hrs])
    srs = np.asarray([cv2.cvtColor(sr, cv2.COLOR_RGB2YCR_CB) for sr in srs])

    # get only the Y channels
    hrs = hrs[:, :, :, 0]
    srs = srs[:, :, :, 0]

    # compute the psrn and ssim scores
    n = hrs.shape[0]
    psnr = np.asarray([peak_signal_noise_ratio(hrs[i], srs[i]) for i in range(n)])
    ssim = np.asarray([structural_similarity(hrs[i], srs[i]) for i in range(n)])

    return psnr, ssim
