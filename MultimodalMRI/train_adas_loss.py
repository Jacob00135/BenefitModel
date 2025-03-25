import torch
from trainer import MRITrainer


class MRITrainerADASLoss(MRITrainer):

    def __init__(self, adas_alpha, *args, **kwargs):
        super(MRITrainerADASLoss, self).__init__(*args, **kwargs)
        self.adas_alpha = adas_alpha

    def loss(self, pred, label, record):
        adas = record['ADAS13'].to(self.device)
        delta = 1e-10

        cn_loss = label[:, 0] * torch.log(pred[:, 0] + delta)
        mci_loss = label[:, 1] * torch.log(pred[:, 1] + delta)
        ad_loss = self.adas_alpha * (1 + adas) * label[:, 2] * torch.log(
            pred[:, 2] + delta)
        loss = torch.sum(cn_loss + mci_loss + ad_loss)
        loss = torch.div(-1 * loss, pred.shape[0])

        return loss


if __name__ == '__main__':
    adas_alpha = 2
    model = MRITrainerADASLoss(
        adas_alpha=adas_alpha,
        model_name='adas_{}'.format(adas_alpha),
        device='cuda:0',
        batch_size=16,
        init_lr=0.001,
        zoom_dataset=True
    )
    model.train()
