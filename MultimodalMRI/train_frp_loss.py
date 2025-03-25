import torch
from trainer import MRITrainer


class MRITrainerFRPLoss(MRITrainer):

    def __init__(self, adas_alpha, *args, **kwargs):
        super(MRITrainerFRPLoss, self).__init__(*args, **kwargs)
        self.adas_alpha = adas_alpha

    def loss(self, pred, label, record):
        adas = record['ADAS13'].to(self.device)
        delta = 1e-10

        ones = torch.ones(pred.shape[0]).to(self.device)
        cn_loss = label[:, 0] * torch.log(pred[:, 0] + delta) * torch.max(
            pred[:, 2] + 0.5, ones)
        mci_loss = label[:, 1] * torch.log(pred[:, 1] + delta) * torch.max(
            pred[:, 2] + 0.5, ones)
        ad_loss = self.adas_alpha * (1 + adas) * label[:, 2] * torch.log(
            pred[:, 2] + delta)
        loss = torch.sum(cn_loss + mci_loss + ad_loss)
        loss = torch.div(-1 * loss, pred.shape[0])

        return loss


if __name__ == '__main__':
    adas_alpha = 1.125
    model = MRITrainerFRPLoss(
        adas_alpha=adas_alpha,
        model_name='frp_{}'.format(adas_alpha),
        device='cuda:0',
        batch_size=16,
        init_lr=0.001,
        zoom_dataset=True
    )
    model.train()
