import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, out_size=3):
        super(MLP, self).__init__()

        fil_num = 100
        drop_rate = 0

        self.dense1 = nn.Linear(in_size, fil_num)
        self.dropout1 = nn.Dropout(drop_rate)
        self.dense2 = nn.Linear(fil_num, out_size)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.softmax1(x)

        return x


if __name__ == '__main__':
    pass
