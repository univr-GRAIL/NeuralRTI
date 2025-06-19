from ignite.metrics import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
import cv2
from skimage.metrics import structural_similarity as ssim

psnr_metric = PeakSignalNoiseRatio()
from PIL import Image
import numpy as np
import cv2 as cv
import torch
from SSIM_PIL import compare_ssim

metric = SSIM(data_range=1.0)


def calc_metrics(gt_path, est_path):
    psnr, ssim = [], []
    for i in range(len(gt_path)):
        image1 = Image.open(gt_path[i])
        image2 = Image.open(est_path[i])
        ssim_val = compare_ssim(image1, image2)
        img1 = cv.imread(est_path[i])
        img2 = cv.imread(gt_path[i])
        psnr_val = cv.PSNR(img1, img2)

        ssim.append(ssim_val)
        psnr.append(psnr_val)

    avg_psnr = np.average(psnr)
    avg_ssim = np.average(ssim)
    return avg_psnr, avg_ssim


def save_model_json(decoder):
    w_list, b_list = [], []
    counter = 0
    for name, param in decoder.named_parameters():

        if name.endswith('weight'):
            weights = param.detach().cpu().numpy()
            weights = weights.T
            while weights.shape[0] % 4 != 0:
                weights = np.concatenate((weights, np.zeros((1, weights.shape[1]), 'float32')), axis=0)

            if counter < 4:
                # weights' columns
                while weights.shape[1] % 4 != 0:
                    weights = np.concatenate((weights, np.zeros((weights.shape[0], 1), 'float32')), axis=1)
            w = np.reshape(weights.T, -1).tolist()
            w = [round(e, 6) for e in w]
            w_list.append(w)

        else:
            biases = param.detach().cpu().numpy()
            if counter < 4:
                # weights' columns
                while biases.shape[0] % 4 != 0:
                    biases = np.concatenate((biases, np.zeros((1,), 'float32')))
            # if '0.weight' in name:
            b = biases.tolist()
            b = [round(e, 6) for e in b]
            b_list.append(b)

        counter += 1

    return w_list, b_list


def calc_metrics_wmask(gt_path, est_path, mask_path):
    psnr_store, ssim_store = [], []
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # A binary mask (0 or 255)
    binary_mask = (mask_img > 0).astype(np.uint8)
    for i in range(len(gt_path)):
        relighted = cv2.imread(est_path[i], cv2.IMREAD_GRAYSCALE)
        groundtruth = cv2.imread(gt_path[i], cv2.IMREAD_GRAYSCALE)
        # Ensure all images are of the same size
        assert relighted.shape == groundtruth.shape == binary_mask.shape, "Images and mask must have the same dimensions."

        # Apply mask to both images

        reference_masked = relighted[binary_mask > 0]
        distorted_masked = groundtruth[binary_mask > 0]

        reference_masked2 = groundtruth * binary_mask
        distorted_masked2 = relighted * binary_mask

        # Compute MSE (Mean Squared Error) over the masked region
        ssim_value, _ = ssim(reference_masked2, distorted_masked2, data_range=255, full=True)
        mse = np.mean((reference_masked - distorted_masked) ** 2)
        if mse == 0:
            return float('inf')  # Return infinity if MSE is zero (images are identical)

        # Compute PSNR
        max_pixel_value = 255.0  # Assume 8-bit images (pixel range 0-255)
        psnr_val = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        psnr_store.append(psnr_val)
        ssim_store.append(ssim_value)
    avg_psnr = np.average(psnr_store)
    avg_ssim = np.average(ssim_store)
    return avg_psnr, avg_ssim
