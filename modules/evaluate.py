import glob
import numpy as np
import pandas as pd
from utils.calculate_psnrWmask import calculate_psnr_with_mask as calcpsnr
from utils.utils import calc_metrics


def evaluate(gt_path, est_path, mask_path):
    results = np.zeros((1, 4))
    gt_images = sorted(glob.glob(gt_path + '/*.jpg'))
    est_images = sorted(glob.glob(est_path + '/*.jpg'))
    psnr_masked, ssim_masked = calcpsnr(gt_images, est_images, mask_path)
    psnr, ssim = calc_metrics(gt_path, est_path)

    results[0, 0] = psnr
    results[0, 1] = psnr_masked
    results[0, 2] = ssim
    results[0, 3] = ssim_masked

    psnr_ssim_result = pd.DataFrame(
        {'PSNR': results[:, 0], 'PSNR-masked': results[:, 1], 'SSIM': results[:, 2],
         'SSIM-masked': results[:, 3]})
    psnr_ssim_result.to_csv(est_path + '/result.csv')
