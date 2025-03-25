import torch
from trainer import ARP2DCNNTrainer
from compute_performance import print_info
from draw_graph import plot


class ARP2DCNNTrainerFRPLossNoADAS(ARP2DCNNTrainer):

    def loss(self, pred, label, adas):
        delta = 1e-10

        ones = torch.ones(pred.shape[0]).to(self.device)
        cn_loss = label[:, 0] * torch.log(pred[:, 0] + delta) * torch.max(pred[:, 1] + 0.5, ones)
        ad_loss = label[:, 1] * torch.log(pred[:, 1] + delta)
        total_loss = torch.div(-1 * torch.sum(cn_loss + ad_loss), pred.shape[0])

        return total_loss


if __name__ == '__main__':
    model_name = 'frp'
    num_epoch = 100
    model = ARP2DCNNTrainerFRPLossNoADAS(
        model_name=model_name,
        device='cuda:0',
        batch_size=16,
        init_lr=0.0001
    )
    model.train(num_epoch)
    print_info([model_name])
    plot(model_name)
