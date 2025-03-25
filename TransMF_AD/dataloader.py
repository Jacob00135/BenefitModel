import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import root_path, zoom_mri_path


class MRIPETDataset(Dataset):

    def __init__(self, dataframe, mri_dir_path):
        super(MRIPETDataset, self).__init__()

        self.data = dataframe
        self.mri_dir_path = mri_dir_path
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        record = self.data.iloc[index, :]

        mri = np.load(os.path.join(self.mri_dir_path, record['filename_MRI']))
        pet = np.load(os.path.join(self.mri_dir_path, record['filename_PET']))
        mri = np.expand_dims(mri, axis=0).astype('float32')
        pet = np.expand_dims(pet, axis=0).astype('float32')
        mri = torch.tensor(mri)
        pet = torch.tensor(pet)

        label = np.zeros(2, dtype='int')
        label[record['COG']] = 1
        label = torch.tensor(label, dtype=torch.float32)

        adas = record['ADAS13']
        benefit = record['benefit']

        return mri, pet, label, adas, benefit


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(root_path, 'datasets/train.csv'))
    row = df[df['filename_MRI'] == '257270.npy']
    df = pd.DataFrame()
    for i in range(10):
        df = pd.concat((df, row))

    dataset = MRIPETDataset(df, zoom_mri_path)
    train_set = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
    for mri, pet, label, adas in train_set:
        print(mri.shape, pet.shape)
