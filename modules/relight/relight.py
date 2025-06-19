import os
import glob
import cv2 as cv
import numpy as np
import pandas as pd
import torch
from modules.utils.params import *
import time
def relight(model_path, gt_path, mask=False):
    input_feature_path = model_path + '/coefficient.npy'
    est_path = model_path + '/relighted'
    if not os.path.exists(est_path):
        os.makedirs(est_path)

    # model = LitAutoEncoder(num_inputs=147)
    [h, w] = cv.imread(model_path + '/plane_0.png', 0).shape
    relighted_img = np.zeros((h * w, 3))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    samples = torch.from_numpy(np.load(input_feature_path)).to(device)
    hw = samples.shape[0]
    if mask:
        mask_img = cv.imread(gt_path + '/mask.png', cv.IMREAD_GRAYSCALE)
        _, binary_mask = cv.threshold(mask_img, 127, 255, cv.THRESH_BINARY)
        mask_img = binary_mask.flatten()
        masked_indices = np.squeeze(np.column_stack(np.where(mask_img == 255)))

    decoder = torch.load(model_path + '/decoder.pth').to(device)
    ld_file = gt_path + '/dirs.lp'
    light_dimension = 2

    def get_lights(ld_file, light_dimension=2):
        with open(ld_file) as f:
            data = f.read()
        data = data.split('\n')
        data = data[
               1:int(data[
                         0]) + 1]  # keep the lines with light directions, remove the first one which is number of samples

        num_lights = len(data)
        ld = np.zeros((num_lights, light_dimension), np.float32)
        for i, dirs in enumerate(data):
            if (len(dirs.split(' ')) == 4):
                sep = ' '
            else:
                sep = '\t'
            s = dirs.split(sep)
            if len(s) == 4:
                ld[i] = [float(s[l]) for l in range(1, light_dimension + 1)]
            else:
                ld[i] = [float(s[l]) for l in range(light_dimension)]
        return ld

    light_dirs = get_lights(ld_file=ld_file, light_dimension=light_dimension)
    for i, l_dir in enumerate(light_dirs):
        lights_list = np.tile(l_dir, reps=hw)
        lights_list = np.reshape(lights_list, (hw, light_dimension))
        light_dir = torch.from_numpy(lights_list)

        # features = torch.from_numpy(unmasked_features.astype('float32'))
        with torch.no_grad():
            zy = torch.cat((samples, light_dir), dim=1).to(device)
            reconst_imgs = decoder(zy.to(device))
            relighted_img = reconst_imgs.cpu().numpy()

            outputs = relighted_img.clip(min=0, max=1)
            if mask:
                relighted_img[masked_indices] = outputs
            else:
                relighted_img = outputs

            outputs = np.reshape(relighted_img, (h, w, 3))
            outputs *= 255
            outputs = outputs.astype('uint8')
            cv.imwrite(est_path + '/relighted' + str(i).zfill(2) + '.jpg', outputs)

    #
# gt_path = sorted(glob.glob(test_dir + '/*.jpg'))
# est_path = sorted(glob.glob(rel_path + '/*.jpg'))
# # est_path_stud = sorted(glob.glob(stud_rel_path + '/*.jpg'))
# # est_path_stud10 = sorted(glob.glob(stud10_rel_path + '/*.jpg'))
# # est_path_reduced = sorted(glob.glob(reduced_rel_path + '/*.jpg'))
# mask_path = test_dir + '/mask.png'
# psnr_masked, ssim_masked = calcpsnr.calculate_psnr_with_mask(gt_path, est_path, mask_path)
# psnr, ssim, _, _ = utils.calc_metrics(gt_path, est_path)
#
# # psnr_stud, ssim_stud, _, _ = utils.calc_metrics(gt_path, est_path_stud)
# # psnr_stud10, ssim_stud10, _, _ = utils.calc_metrics(gt_path, est_path_stud10)
# # psnr_reduced, ssim_reduced, _, _ = utils.calc_metrics(gt_path, est_path_reduced)
#
# results[0, 0] = psnr
# results[0, 1] = psnr
# results[0, 2] = psnr
# results[0, 3] = psnr_masked
# results[0, 4] = ssim_masked
# results[0, 5] = ssim
# results[0, 6] = ssim
# results[0, 7] = ssim
#
# psnr_ssim_result = pd.DataFrame(
#     {'PSNR-reduced': results[:, 0], 'PSNR-student10': results[:, 1], 'PSNR-student': results[:, 2],
#      'PSNR-masked': results[:, 3], 'SSIM-masked': results[:, 4], 'SSIM-student10': results[:, 5],
#      'SSIM-student': results[:, 6], 'SSIM-teacher': results[:, 7]})
# psnr_ssim_result.to_csv(datasetname + '/' + datasetname + f'_result(DiskNeural_orig).csv')
