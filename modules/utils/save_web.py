import os
import numpy as np
import torch
from utils.utils import save_model_json
import json


def save_web_format(decoder_fpath, coeff_fpath, h, w, comp_coeff, json_dir, samples):
    decoder_path = torch.load(decoder_fpath, map_location='cuda:0')
    features = np.load(coeff_fpath)

    w_list, b_list = save_model_json(decoder_path)
    max_f = [float(np.max(features[:, i])) for i in range(comp_coeff)]
    min_f = [float(np.min(features[:, i])) for i in range(comp_coeff)]
    info = {
        'height': h,
        'width': w,
        'planes': comp_coeff,
        'samples': samples,
        'format': 'jpg',
        'quality': 95,
        'type': 'neural',
        'colorspace': 'bgr',
        'max': max_f,
        'min': min_f,
        'weights': w_list,
        'biases': b_list

    }

    with open(json_dir + '/info.json', 'w') as f:
        json.dump(info, f)
