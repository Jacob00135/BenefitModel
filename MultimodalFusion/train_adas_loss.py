import torch
from trainer import FusionTrainer
from draw_graph import plot


class FusionTrainerADASLoss(FusionTrainer):

    def __init__(self, adas_alpha, *args, **kwargs):
        super(FusionTrainerADASLoss, self).__init__(*args, **kwargs)
        self.adas_alpha = adas_alpha

    def loss(self, pred, label, adas):
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
    num_epoch = 150
    model_name = 'adas_{}'.format(adas_alpha)
    model = FusionTrainerADASLoss(
        adas_alpha=adas_alpha,
        model_name=model_name,
        device='cpu',
        init_lr=0.05
    )
    model.train(num_epoch)
    plot(model_name, xlim=num_epoch)

