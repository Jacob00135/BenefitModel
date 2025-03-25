import os
import torch
import numpy as np
from torch.utils.data import Dataset


class MRI3DDataset(Dataset):

    def __init__(self, dataframe, mri_dir_path=None):
        self.data = dataframe
        self.mri_dir_path = mri_dir_path
        if self.mri_dir_path is None:
            self.mri_dir_path = mri_path

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        record = self.data.iloc[idx, :]
        
        filename = record['filename']
        filepath = os.path.join(self.mri_dir_path, filename)
        img = np.load(filepath).astype('float32')
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img)

        label = np.zeros(3, dtype='int')
        label[record['COG']] = 1
        label = torch.tensor(label)

        extra_record = record[['ADAS13', 'benefit']].to_dict()

        return img, label, extra_record
