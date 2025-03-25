import os
import torch
import numpy as np
import pandas as pd
from config import root_path, mri_path
from torch.utils.data import Dataset, DataLoader


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


if __name__ == '__main__':
    data_path = os.path.join(root_path, 'datasets/data.csv')
    df = pd.read_csv(data_path)
    df = df[df['filename'] == os.listdir(mri_path)[0]]
    for i in range(16):
        df = pd.concat((df, df.iloc[0:1, :]))

    dataset = MRI3DDataset(df)
    dataset = DataLoader(dataset, batch_size=4, drop_last=False)
    for img, label, record in dataset:
        break
