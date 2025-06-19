import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_psnr_with_mask(gt_path, est_path, mask_path):
    psnr, ssim_store = [], []
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # A binary mask (0 or 255)
    mask = (mask_img > 0).astype(np.uint8)

    for i in range(len(gt_path)):
        relighted = cv2.imread(est_path[i], cv2.IMREAD_GRAYSCALE)
        groundtruth = cv2.imread(gt_path[i], cv2.IMREAD_GRAYSCALE)
        # Ensure all images are of the same size
        assert relighted.shape == groundtruth.shape == mask.shape, "Images and mask must have the same dimensions."

        # Apply mask to both images

        reference_masked = relighted[mask > 0]
        distorted_masked = groundtruth[mask > 0]

        reference_masked2 = groundtruth * mask
        distorted_masked2 = relighted * mask

        # Compute MSE (Mean Squared Error) over the masked region
        ssim_value, _ = ssim(reference_masked2, distorted_masked2, data_range=255,full=True)
        mse = np.mean((reference_masked - distorted_masked) ** 2)
        if mse == 0:
            return float('inf')  # Return infinity if MSE is zero (images are identical)

        # Compute PSNR
        max_pixel_value = 255.0  # Assume 8-bit images (pixel range 0-255)
        psnr_val = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        psnr.append(psnr_val)
        ssim_store.append(ssim_value)
    avg_psnr = np.average(psnr)
    avg_ssim = np.average(ssim_store)
    return avg_psnr, avg_ssim
