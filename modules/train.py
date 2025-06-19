import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(logging_interval='epoch')
import cv2 as cv
import os
from utils.params import *
from dataset.custom_dataset import Mycustomdataset
from model.neural_model import LitAutoEncoder
from utils.save_web import save_web_format
from dataset.datasetwmask import MLIC
import torch.nn
import numpy as np
from torch.utils.data import DataLoader
import time
import warnings

warnings.filterwarnings("ignore")

t1 = time.time()
#####################################################

teacher_spath = output_path + '/Model_Files'  # folder for saving teacher model and feature planes.

if not os.path.exists(teacher_spath):
    os.makedirs(teacher_spath, mode=0o777, exist_ok=True)
mlic = MLIC(data_path=data_path, ld_file=ld_file, src_img_type=src_img_type, mask=mask)

##########################
h, w = mlic.h, mlic.w
unmasked_features = np.zeros((h * w, 9))

hw, num_samples, no_channels = mlic.samples.shape
sample_size = int(hw * num_samples)
limit = hw
samples = np.reshape(mlic.samples, (limit, -1))

input_samples = torch.from_numpy(samples)
np.random.seed(seed=42)
all_indices = np.random.choice(sample_size, sample_size, replace=False)

p_idx = all_indices % (hw)
gt_idx = (all_indices * num_samples // sample_size).astype(np.uint8)
###########################################################################
pixel_index = np.zeros((sample_size, 2), dtype=int)
pixel_index[:, 0] = p_idx
pixel_index[:, 1] = gt_idx
root_dir = mlic
# ------ prepare custom MLIC dataset for dataloader
custom_dataset = Mycustomdataset(csv_file=pixel_index, info=[hw, num_samples, no_channels],
                                 data=[mlic.samples, mlic.ld])
train_set, valid_set = torch.utils.data.random_split(custom_dataset, [0.9, 0.1])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)
if mask:
    mask_img = mlic.binary_mask.flatten()
    masked_indices = np.squeeze(np.column_stack(np.where(mask_img == 255)))

del mlic, root_dir, samples, custom_dataset
trainer = L.Trainer(enable_model_summary=False, max_epochs=max_epochs, check_val_every_n_epoch=1,
                    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
model = LitAutoEncoder(num_inputs=num_samples * no_channels)
# print(model.decoder)
print("Training NeuralRTI")
trainer.fit(model, train_loader, valid_loader)
encoder = model.encoder
decoder = model.decoder

decoder_fpath = teacher_spath + "/model.pth"
coeff_fpath = teacher_spath + '/coefficient.npy'
torch.save(decoder, decoder_fpath)
encoder.eval()
with torch.no_grad():
    reconst_imgs = encoder(input_samples)
features = reconst_imgs.cpu().numpy()
np.save(coeff_fpath, features)

max_f = [float(np.max(features[:, i])) for i in range(comp_coeff)]
min_f = [float(np.min(features[:, i])) for i in range(comp_coeff)]
bit_feat = 8
for i in range(comp_coeff):
    features[:, i] = np.interp(features[:, i], (min_f[i], max_f[i]), (0, 2 ** bit_feat - 1))

# reshape features to store them as images and do it
if mask:
    unmasked_features[masked_indices] = features
else:
    unmasked_features = features
features = np.reshape(unmasked_features, (h, w, comp_coeff))

for j in range(comp_coeff // 3):
    cv.imwrite(teacher_spath + '/plane' + '_' + str(j) + '.jpg',
               features[..., 3 * j:3 * (j + 1)].astype(np.uint8))
    cv.imwrite(teacher_spath + '/plane' + '_' + str(j) + '.png',
               features[..., 3 * j:3 * (j + 1)].astype(np.uint8))

save_web_format(decoder_fpath, coeff_fpath, h, w, comp_coeff, teacher_spath, num_samples)
t2 = time.time()
print('done!')
print(f'--- {int(t2 - t1) // 60 // 60} h {int(t2 - t1) // 60 % 60} m {int(t2 - t1) % 60} s ---')
