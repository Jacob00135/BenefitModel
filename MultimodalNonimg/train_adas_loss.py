import torch
from trainer import NonimgTrainer
from draw_graph import plot


class NonimgTrainerADASLoss(NonimgTrainer):

    def __init__(self, adas_alpha, *args, **kwargs):
        super(NonimgTrainerADASLoss, self).__init__(*args, **kwargs)
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
    model_name = 'adas_{}'.format(adas_alpha)
    model = NonimgTrainerADASLoss(
        adas_alpha=adas_alpha,
        model_name=model_name,
        device='cpu',
        init_lr=0.01
    )
    model.train()
    plot(model_name)

