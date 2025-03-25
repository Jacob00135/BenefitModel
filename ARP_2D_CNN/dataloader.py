import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import root_path, mri_2d_path


class MRI2DDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        record = self.data.iloc[index, :]

        filename = record['filename']
        filepath = os.path.join(mri_2d_path, filename)
        image = (np.load(filepath) / 255).astype('float32')
        image = torch.tensor(image)

        label = np.zeros(2, dtype='int')
        label[record['COG']] = 1
        label = torch.tensor(label)

        adas = record['ADAS13']
        benefit = record['benefit']

        return image, label, adas, benefit


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_set = pd.read_csv(os.path.join(root_path, 'datasets/train.csv'))
    train_set = MRI2DDataset(train_set)
    train_set = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=False)
    for batch_image, batch_label, adas, benefit in train_set:
        breakpoint()
        break
