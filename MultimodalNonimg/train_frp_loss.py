import torch
from trainer import NonimgTrainer
from draw_graph import plot
from compute_performance import print_info


class NonimgTrainerFRPLoss(NonimgTrainer):

    def __init__(self, adas_alpha, *args, **kwargs):
        super(NonimgTrainerFRPLoss, self).__init__(*args, **kwargs)
        self.adas_alpha = adas_alpha

    def loss(self, pred, label, adas):
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
    adas_alpha = 2.5
    model_name = 'frp_{}'.format(adas_alpha)
    model = NonimgTrainerFRPLoss(
        adas_alpha=adas_alpha,
        model_name=model_name,
        device='cpu',
        init_lr=0.01
    )
    model.train()
    model_name_list = [
        'src',
        'adas_2',
        'frp',
        model_name
    ]
    print_info(model_name_list)
    plot(model_name)
