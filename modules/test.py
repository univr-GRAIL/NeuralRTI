import glob
import numpy as np
import pandas as pd
from relight.relight import relight
from utils.utils import calc_metrics, calc_metrics_wmask

results = np.zeros((1, 2))
masked = True
model_path = 'C:/Users/Utente/OneDrive/Desktop/test_dataset/outputs/Teacher'
gt_path = 'C:/Users/Utente/OneDrive/Desktop/test_dataset/Test'
est_path = 'C:/Users/Utente/OneDrive/Desktop/test_dataset/outputs/Teacher/relighted'
relight(model_path, gt_path)
