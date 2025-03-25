import os
import scipy
import numpy as np
from time import time as get_timestamp
from tqdm import tqdm

mri_path = '/home/xjy/ADNI_MRI_npy'
dst_path = '/home/xjy/python_code/BenefitModel/zoom_MRI_npy'
zoom_factors = (0.1, 0.1, 0.2)
total_set = set(os.listdir(mri_path))
zoom_set = set(os.listdir(dst_path))
for fn in tqdm(total_set - zoom_set):
    fp = os.path.join(mri_path, fn)
    mri = np.load(fp) / 8
    mri = scipy.ndimage.zoom(mri, zoom=zoom_factors, order=1)

    save_path = os.path.join(dst_path, fn)
    np.save(save_path, mri)
