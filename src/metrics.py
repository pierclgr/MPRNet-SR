import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import math


def compute_metrics(hr, sr):
    # clip to 0-1 ranges
    sr = np.clip(sr, a_max=1, a_min=0)
    hr = np.clip(hr, a_max=1, a_min=0)

    # convert to 0-255 range
    sr *= 255
    hr *= 255
    sr = sr.astype(np.uint8)
    hr = hr.astype(np.uint8)

    # convert the two images from rgb to YCrCb
    hr = cv2.cvtColor(hr, cv2.COLOR_RGB2YCR_CB)
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2YCR_CB)

    # get only the Y channels of the images
    hr = hr[:, :, 0]
    sr = sr[:, :, 0]

    # compute the psrn and ssim scores
    psnr = peak_signal_noise_ratio(hr, sr)
    ssim = structural_similarity(hr, sr)

    return psnr, ssim
