import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pooling, drop_rate, BN=True, relu_type='leaky'):
        super(ConvLayer, self).__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type=='leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class _CNN_Bone_Zoom(nn.Module):
    def __init__(self):
        super(_CNN_Bone_Zoom, self).__init__()
        num = 20
        p = 0.37
        self.block1 = ConvLayer(1, num, (3, 1, 0), (2, 2, 0), p)
        self.block2 = ConvLayer(num, 2 * num, (3, 1, 0), (2, 2, 0), p)
        self.size = self.test_size()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def test_size(self):
        case = torch.ones((1, 1, 18, 22, 36))
        output = self.forward(case)
        return output.shape[1]


class MRIMLP(nn.Module):
    def __init__(self, in_size):  # if binary out_size=2; trinary out_size=3
        super(MRIMLP, self).__init__()
        fil_num = 100
        drop_rate = 0.5
        out_size = 3
        self.fil_num = fil_num
        self.out_size = out_size
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size, fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, out_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x, get_intermediate_score=False):
        x = self.dense1(x)
        if get_intermediate_score:
            return x
        x = self.dense2(x)
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.dense1[1].weight.view(self.fil_num, self.in_size//(6*6*6), 6, 6, 6)
        B = fcn.dense2[2].weight.view(self.out_size, self.fil_num, 1, 1, 1)
        C = fcn.dense1[1].bias
        D = fcn.dense2[2].bias
        fcn.dense1[1] = nn.Conv3d(self.in_size//(6*6*6), self.fil_num, 6, 1, 0).cuda()
        fcn.dense2[2] = nn.Conv3d(self.fil_num, self.out_size, 1, 1, 0).cuda()
        fcn.dense1[1].weight = nn.Parameter(A)
        fcn.dense2[2].weight = nn.Parameter(B)
        fcn.dense1[1].bias = nn.Parameter(C)
        fcn.dense2[2].bias = nn.Parameter(D)
        return fcn


class FusionMLP(nn.Module):
    def __init__(self, in_size, out_size=3):
        super(FusionMLP, self).__init__()

        fil_num = 100
        drop_rate = 0.5

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
    model = _CNN_Bone_Zoom()
    print(model)
