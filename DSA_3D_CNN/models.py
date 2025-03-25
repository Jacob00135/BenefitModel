import torch
from torch import nn


class DSA_3D_CNN(nn.Module):
    def __init__(self):
        super(DSA_3D_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        )

        # 计算输入形状
        # (110, 110, 110) --> (110, 110, 8) --> (55, 55, 8)
        # (55, 55, 8) --> (55, 55, 8) --> (27, 27, 8)
        # (27, 27, 8) --> (27, 27, 8) --> (13, 13, 8)
        linear_shape = (13, 13, 13)
        units = 8 * linear_shape[0] * linear_shape[1] * linear_shape[2]

        self.fc1 = nn.Sequential(
            nn.Linear(units, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU()
        )

        self.out = nn.Linear(500, 3)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        x = nn.Dropout(0.8)(x)
        x = self.out(x)
        prob = self.out_act(x)
        return prob


class DSA_3D_CNN_Zoom(nn.Module):
    def __init__(self):
        super(DSA_3D_CNN_Zoom, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        )

        """
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        )
        """

        # 计算输入形状
        # (18, 22, 36) --> (18, 22, 36) --> (9, 11, 18)
        # (9, 11, 18) --> (9, 11, 18) --> (4, 5, 9)
        # (4, 5, 9) --> (4, 5, 9) --> (2, 2, 4)
        units = 8 * 4 * 5 * 9

        self.fc1 = nn.Sequential(
            nn.Linear(units, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU()
        )

        self.out = nn.Linear(500, 3)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        x = nn.Dropout(0.8)(x)
        x = self.out(x)
        prob = self.out_act(x)
        return prob


if __name__ == '__main__':
    # 测试(110, 110, 110)
    """
    conv_form = (w + 2 * padding - kernel_size) / stride + 1
    pool_form = (w - kernel_size) / stride + 1

    import os
    import numpy as np
    from skimage import transform

    mri_path = '/home/xjy/ADNI_MRI_npy/'
    batch_image = np.zeros((4, 1, 110, 110, 110), dtype='float32')
    for i, fn in enumerate(os.listdir(mri_path)):
        img = np.load(os.path.join(mri_path, fn))
        img = transform.resize(img, (110, 110, 110), order=1, preserve_range=True)
        batch_image[i] = np.expand_dims(img, 0)
        if i >= 3:
            break
    batch_image = torch.tensor(batch_image)

    model = DSA_3D_CNN()
    prediction = model(batch_image)
    print(prediction)
    """
