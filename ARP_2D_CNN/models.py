import torch
import pandas as pd
from torch.nn import (
    Module, Sequential, Conv2d, ReLU, Softmax, Linear, BatchNorm1d, Dropout
)
from torchvision.models import vgg11


class att(Module):

    def __init__(self, input_channel):
        "the soft attention module"
        super(att,self).__init__()

        self.conv1 = Sequential(
            Conv2d(
                in_channels=input_channel,
                out_channels=512,
                kernel_size=1
            ), 
            ReLU()
        )
        self.conv2 = Sequential(
            Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1
            ), 
            ReLU()
        )
        self.conv3 = Sequential(
            Conv2d(
                in_channels=256,
                out_channels=64,
                kernel_size=1
            ),
            ReLU()
        )
        self.conv4 = Sequential(
            Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=1
            ),
            Softmax(dim=2)
        )

    def forward(self, x):
        mask = x
        mask = self.conv1(mask)
        mask = self.conv2(mask)
        mask = self.conv3(mask)
        att = self.conv4(mask)
        output = torch.mul(x, att)
        return output


class CNN(Module):

    def __init__(self, drop_prob=0.5):
        
        super(CNN, self).__init__()

        self.drop_prob = drop_prob

        # Feature Extraction
        self.ft_ext = vgg11(pretrained=True)
        self.ft_ext_modules = list(self.ft_ext.children())[0][:19] # remove the Maxpooling layer
        self.ft_ext = Sequential(*self.ft_ext_modules)
        for p in self.ft_ext.parameters():
            p.requires_grad = True
        # 224 --> 112 --> 56 --> 28 --> 14

        # Classifier
        feature_shape=(512, 14, 14)
        fc1_input_features = feature_shape[0] * feature_shape[1] * feature_shape[2]
        fc1_output_features = feature_shape[0] * 2
        fc2_output_features = int(fc1_output_features / 4)
        self.attn = att(feature_shape[0])
        self.fc1 = Sequential(
             Linear(fc1_input_features, fc1_output_features),
             BatchNorm1d(fc1_output_features),
             ReLU()
        )
        self.fc2 = Sequential(
             Linear(fc1_output_features, fc2_output_features),
             BatchNorm1d(fc2_output_features),
             ReLU()
        )

        # self.out = Linear(fc2_output_features, 2)
        self.out = Sequential(
            Linear(fc2_output_features, 2),
            Softmax(dim=1)
        )

    def forward(self, x):
        x = self.ft_ext(x)
        x = self.attn(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = Dropout(self.drop_prob)(x)
        x = self.fc2(x)
        x = Dropout(self.drop_prob)(x)
        prob = self.out(x) 

        return prob


if __name__ == '__main__':
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    from config import root_path
    from dataloader import MRI2DDataset

    train_set = pd.read_csv(os.path.join(root_path, 'datasets/train.csv'))
    train_set = MRI2DDataset(train_set)
    train_set = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=False)
    for batch_image, batch_label, adas, benefit in train_set:
        break

    net = CNN()
    result = net(batch_image)
    print(result.shape)
    print(result)
