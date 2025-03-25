import torch
from trainer import ARP2DCNNTrainer
from compute_performance import print_info
from draw_graph import plot


class ARP2DCNNTrainerADASLoss(ARP2DCNNTrainer):

    def __init__(self, adas_alpha, *args, **kwargs):
        super(ARP2DCNNTrainerADASLoss, self).__init__(*args, **kwargs)
        self.adas_alpha = adas_alpha

    def loss(self, pred, label, adas):
        delta = 1e-10
        cn_loss = label[:, 0] * torch.log(pred[:, 0] + delta)
        ad_loss = self.adas_alpha * (1 + adas) * label[:, 1] * torch.log(pred[:, 1] + delta)
        total_loss = torch.div(-1 * torch.sum(cn_loss + ad_loss), pred.shape[0])

        return total_loss


if __name__ == '__main__':
    adas_alpha = 2
    model_name = 'adas_{}'.format(adas_alpha)
    num_epoch = 100
    model = ARP2DCNNTrainerADASLoss(
        adas_alpha=adas_alpha,
        model_name=model_name,
        device='cuda:0',
        batch_size=16,
        init_lr=0.0001
    )
    model.train(num_epoch)
    print_info([model_name])
    plot(model_name)
