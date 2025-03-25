import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply


class sNet(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Conv3d(dim // 2, dim // 1, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim // 1),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim // 1, dim * 2, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(dim * 2),
            nn.LeakyReLU(),
            nn.Conv3d(dim * 2, dim, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(),
            nn.AvgPool3d(2, stride=2)
        )

        """
        conv1: 32, kernel_size=3, padding=1, stride=1
        pool1: kernel_size=2, stride=2

        conv2.1: 32, kernel_size=3, padding=1, stride=1
        conv2.2: 64, kernel_size=3, padding=1, stride=1
        pool2: kernel_size=2, stride=2

        conv3.1: 64, kernel_size=3, padding=1, stride=1
        conv3.2: 128, kernel_size=3, padding=1, stride=1
        pool3: kernel_size=2, stride=2

        conv4.1: 256, kernel_size=3, padding=1, stride=1
        conv4.2: 128, kernel_size=1, padding=0, stride=1
        pool4: kernel_size=2, stride=2
        """

        """
        (b, 1, 182, 218, 182)

        conv1: (b, 32, 182, 218, 182)
        pool1: (b, 32, 91, 109, 91)

        conv2.1: (b, 32, 91, 109, 91)
        conv2.2: (b, 64, 91, 109, 91)
        pool2: (b, 64, 45, 54, 45)

        conv3.1: (b, 64, 45, 54, 45)
        conv3.2: (b, 128, 45, 54, 45)
        pool3: (b, 128, 22, 27, 22)

        conv4.1: (b, 256, 22, 27, 22)
        conv4.2: (b, 128, 22, 27, 22)
        pool4: (b, 128, 11, 13, 11)
        """


    def forward(self, mri):
        conv1_out = self.conv1(mri)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)

        return conv4_out


class sNetZoom(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        """
        conv1: (18, 22, 36) --> (16, 20, 34) --> (8, 10, 17)
        conv2: (8, 10, 17) --> (6, 8, 15) --> (3, 4, 7)
        """

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 2, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, stride=2)
        )

    def forward(self, mri):
        conv1_out = self.conv1(mri)
        conv2_out = self.conv2(conv1_out)

        return conv2_out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _ = x.shape
        h = self.heads
        context = default(context, x)

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return self.norm(x)


class CrossTransformer_MOD_AVG(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout),
                Transformer(dim, 1, heads, dim_head, mlp_dim, dropout=dropout)
            ]))
        self.gap = nn.Sequential(Rearrange('b n d -> b d n'),
                                 nn.AdaptiveAvgPool1d(1),
                                 Rearrange('b d n -> b (d n)'))
        self.gmp = nn.Sequential(Rearrange('b n d -> b d n'),
                                 nn.AdaptiveMaxPool1d(1),
                                 Rearrange('b d n -> b (d n)'))


    def forward(self, mri_tokens, pet_tokens):
        for mri_enc, pet_enc in self.layers:
            mri_tokens = mri_enc(mri_tokens, context=pet_tokens) + mri_tokens
            pet_tokens = pet_enc(pet_tokens, context=mri_tokens) + pet_tokens

        mri_cls_avg = self.gap(mri_tokens)
        mri_cls_max = self.gmp(mri_tokens)
        pet_cls_avg = self.gap(pet_tokens)
        pet_cls_max = self.gmp(pet_tokens)
        cls_token = torch.cat([mri_cls_avg,  pet_cls_avg, mri_cls_max, pet_cls_max], dim=1)
        return cls_token


class ModelAD(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.mri_cnn = sNet(dim)
        self.pet_cnn = sNet(dim)
        self.fuse_transformer = CrossTransformer_MOD_AVG(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_cls = nn.Sequential(nn.Linear(dim * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(64, 2), nn.Softmax(dim=1))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool3d(1), Rearrange('b c x y z -> b (c x y z)'))
        self.D = nn.Sequential(nn.Linear(dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mri, pet):
        # forward CNN
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)

        alpha = torch.Tensor([2]).to(mri.device)
        mri_embedding_vec = revgrad(self.gap(mri_embeddings), alpha)
        pet_embedding_vec = revgrad(self.gap(pet_embeddings), alpha)

        # forward discriminator
        D_MRI_logits = self.D(mri_embedding_vec)
        D_PET_logits = self.D(pet_embedding_vec)

        # forward cross transformer
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        # 至此，形状已变为：(batch_size, self.dim * 4)
        output_logits = self.fc_cls(output_pos)

        return output_logits, D_MRI_logits, D_PET_logits


class ModelADZoom(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.mri_cnn = sNetZoom(dim)
        self.pet_cnn = sNetZoom(dim)
        self.fuse_transformer = CrossTransformer_MOD_AVG(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_cls = nn.Sequential(
            nn.Linear(dim * 4, dim * 4),
            nn.BatchNorm1d(dim * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim * 4, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool3d(1), Rearrange('b c x y z -> b (c x y z)'))
        self.D = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, mri, pet):
        # forward CNN
        mri_embeddings = self.mri_cnn(mri)  # shape (b, d, x, y, z,)
        pet_embeddings = self.pet_cnn(pet)  # shape (b, d, x, y, z,)

        alpha = torch.Tensor([2]).to(mri.device)
        mri_embedding_vec = revgrad(self.gap(mri_embeddings), alpha)
        pet_embedding_vec = revgrad(self.gap(pet_embeddings), alpha)

        # forward discriminator
        D_MRI_logits = self.D(mri_embedding_vec)
        D_PET_logits = self.D(pet_embedding_vec)

        # forward cross transformer
        mri_embeddings = rearrange(mri_embeddings, 'b d x y z -> b (x y z) d')
        pet_embeddings = rearrange(pet_embeddings, 'b d x y z -> b (x y z) d')
        output_pos = self.fuse_transformer(mri_embeddings, pet_embeddings)  # shape (b, xyz, d)
        # 至此，形状已变为：(batch_size, self.dim * 4)
        output_logits = self.fc_cls(output_pos)

        return output_logits, D_MRI_logits, D_PET_logits


if __name__ == '__main__':
    import os
    import pandas as pd
    from torch.utils.data import DataLoader
    from config import root_path, zoom_mri_path, mri_path
    from dataloader import MRIPETDataset

    # 载入数据
    test_mapping = {'train': '257270.npy', 'test': '55885.npy'}
    dataset_list = []
    dataloader_list = []
    for k, v in test_mapping.items():
        df = pd.read_csv(os.path.join(root_path, 'datasets/{}.csv'.format(k)))
        row = df[df['filename_MRI'] == v]
        df = pd.DataFrame()
        for i in range(10):
            df = pd.concat((df, row))

        dataset = MRIPETDataset(df, zoom_mri_path)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
        dataset_list.append(dataset)
        dataloader_list.append(data_loader)
    train_set, test_set = dataset_list
    train_loader, test_loader = dataloader_list

    # 载入模型
    dim = 32
    net = ModelADZoom(
        dim=dim,  # 128
        depth=3,
        heads=4,
        dim_head=dim // 4,  # 32
        mlp_dim=dim * 4,  # 512
        dropout=0
    )
    optim = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0)
    loss = torch.nn.CrossEntropyLoss()

    # 模拟训练
    """
    mri或pet: sNet(dim) --> nn.AdaptiveAvgPool3d(1) --> flatten --> Linear(dim, 128) --> Linear(128, 2)
    output: sNet(dim) --> mri_embeddings, pet_embeddings -->
            CrossTransformer_MOD_AVG(dim, depth, heads, dim_head, mlp_dim, dropout)(mri_embeddings, pet_embeddings) -->
            Linear(dim * 4, 512) --> Linear(512, 64) --> Linear(64, 2)
    """
    net.train(True)
    for mri, pet, label, adas, benefit in train_loader:
        optim.zero_grad()
        output, mri_logits, pet_logits = net(mri, pet)
        ce_loss = loss(output, label)
        mri_gt = torch.zeros((mri_logits.shape[0], 2), dtype=torch.float32)
        mri_gt[:, 1] = 1
        pet_gt = torch.zeros((pet_logits.shape[0], 2), dtype=torch.float32)
        pet_gt[:, 0] = 1
        ad_loss = (loss(mri_logits, mri_gt) + loss(pet_logits, pet_gt)) / 2
        all_loss = ce_loss + ad_loss
        all_loss.backward()
        optim.step()
