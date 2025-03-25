import torch
from trainer import TransMFTrainer
from compute_performance import print_info
from draw_graph import plot


class TransMFTrainerFRPLossNoADAS(TransMFTrainer):

    def frp_loss(self, label, pred):
        delta = 1e-10

        ones = torch.ones(pred.shape[0]).to(self.device)
        cn_loss = label[:, 0] * torch.log(pred[:, 0] + delta) * torch.max(pred[:, 1] + 0.5, ones)
        ad_loss = label[:, 1] * torch.log(pred[:, 1] + delta)
        total_loss = torch.div(-1 * torch.sum(cn_loss + ad_loss), pred.shape[0])

        return total_loss

    def loss(self, label, adas, pred, mri_pred, pet_pred):
        mri_gt = torch.zeros((mri_pred.shape[0], 2), dtype=torch.float32).to(self.device)
        mri_gt[:, 1] = 1
        pet_gt = torch.zeros((pet_pred.shape[0], 2), dtype=torch.float32).to(self.device)
        pet_gt[:, 0] = 1

        loss_1 = self.frp_loss(label, pred)
        loss_2 = self.frp_loss(mri_gt, mri_pred)
        loss_3 = self.frp_loss(pet_gt, pet_pred)

        all_loss = loss_1 + (loss_2 + loss_3) / 2

        return all_loss


if __name__ == '__main__':
    model_name = 'frp'
    num_epoch = 100
    model = TransMFTrainerFRPLossNoADAS(
        model_name=model_name,
        device='cuda:0',
        batch_size=16,
        init_lr=0.0001,
        zoom_dataset=True
    )
    model.train(num_epoch)
    print_info([model_name])
    plot(model_name)
