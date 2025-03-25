import os
import random
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torch.optim import Adam
from config import root_path, zoom_mri_path, mri_path
from models import ModelAD, ModelADZoom
from dataloader import MRIPETDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def one_hot(array, num_classes):
    one_hot_array = np.zeros((array.shape[0], num_classes), dtype=array.dtype)
    for i in range(array.shape[0]):
        one_hot_array[i, array[i]] = 1

    return one_hot_array


class TransMFTrainer(object):

    def __init__(self,
        model_name, device='cpu', batch_size=16, init_lr=0.0001,
        zoom_dataset=False
    ):
        setup_seed(1124)

        # 初始化
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.num_classes = 2
        self.training_performance = {}
        self.zoom_dataset = zoom_dataset
        generate_dir_list = [
            os.path.join(root_path, 'checkpoints', self.model_name),
            os.path.join(root_path, 'eval_result', self.model_name),
            os.path.join(root_path, 'predict_result', self.model_name)
        ]
        for p in generate_dir_list:
            if not os.path.exists(p):
                os.mkdir(p)
        print('初始化完毕')

        # 载入数据
        if self.zoom_dataset:
            self.mri_dir_path = zoom_mri_path
        else:
            self.mri_dir_path = mri_path
        train_set = pd.read_csv(os.path.join(root_path, 'datasets/train.csv'))

        # region 测试用代码
        """
        row = train_set[train_set['filename_MRI'] == '257270.npy']
        train_set = pd.DataFrame()
        for i in range(2):
            train_set = pd.concat((train_set, row))
        """
        # endregion

        train_dataset = MRIPETDataset(train_set, self.mri_dir_path)
        self.train_set = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        print('载入数据集完毕')

        # 载入模型
        if self.zoom_dataset:
            dim = 32
            self.net = ModelADZoom(
                dim=dim,
                depth=3,
                heads=4,
                dim_head=dim // 4,
                mlp_dim=dim * 4,
                dropout=0.5
            )
        else:
            dim = 128
            self.net = ModelAD(
                dim=dim,
                depth=3,
                heads=4,
                dim_head=dim // 4,
                mlp_dim=dim * 4,
                dropout=0
            )
        self.net = self.net.to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.init_lr, weight_decay=0)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.98)
        print('载入模型完毕')

    @staticmethod
    def cross_entropy_loss(pred, label):
        delta = 1e-10

        return torch.div(
            -1 * torch.sum(label * torch.log(pred + delta)),
            pred.shape[0]
        )

    def loss(self, label, adas, pred, mri_pred, pet_pred):
        ce_loss = self.cross_entropy_loss(pred, label)
        mri_gt = torch.zeros((mri_pred.shape[0], 2), dtype=torch.float32).to(self.device)
        mri_gt[:, 1] = 1
        pet_gt = torch.zeros((pet_pred.shape[0], 2), dtype=torch.float32).to(self.device)
        pet_gt[:, 0] = 1
        ad_loss = (self.cross_entropy_loss(mri_pred, mri_gt) + self.cross_entropy_loss(pet_pred, pet_gt)) / 2
        all_loss = ce_loss + ad_loss

        return all_loss

    def train_an_epoch(self):
        self.net.train(True)
        for mri, pet, label, adas, benefit in self.train_set:
            mri = mri.to(self.device)
            pet = pet.to(self.device)
            label = label.to(self.device)
            adas = adas.to(self.device)

            self.optim.zero_grad()
            pred, mri_pred, pet_pred = self.net(mri, pet)
            loss = self.loss(label, adas, pred, mri_pred, pet_pred)
            loss.backward()
            self.optim.step()

    def save_model(self, index):
        path = os.path.join(
            root_path,
            'checkpoints',
            self.model_name,
            '{}.pth'.format(index)
        )
        torch.save(self.net.state_dict(), path)

    def load_model(self, index):
        backbone_path = os.path.join(
            root_path,
            'checkpoints',
            self.model_name,
            'backbone_{}.pth'.format(index)
        )
        backbone_state_dict = torch.load(backbone_path, map_location=self.device)
        self.backbone.load_state_dict(backbone_state_dict)

        mlp_path = os.path.join(
            root_path,
            'checkpoints',
            self.model_name,
            'mlp_{}.pth'.format(index)
        )
        mlp_state_dict = torch.load(mlp_path, map_location=self.device)
        self.mlp.load_state_dict(mlp_state_dict)

    def predict(self, stage):
        # 载入数据
        df = pd.read_csv(os.path.join(root_path, 'datasets/{}.csv'.format(stage)))

        # region 测试用代码
        """
        test_mapping = {'train': '257270.npy', 'test': '55885.npy'}
        row = df[df['filename_MRI'] == test_mapping[stage]]
        df = pd.DataFrame()
        for i in range(2):
            df = pd.concat((df, row))
        """
        # endregion

        dataset = MRIPETDataset(df, self.mri_dir_path)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False
        )

        # 初始化变量
        num_sample = len(dataset)
        prediction = np.zeros((num_sample, self.num_classes), dtype='float32')
        labels = np.zeros(num_sample, dtype='int32')
        benefit_array = np.zeros(num_sample, dtype='float32')
        loss_array = np.zeros(num_sample, dtype='float32')

        # 预测
        self.net.eval()
        with torch.no_grad():
            for i, (mri, pet, label, adas, benefit) in enumerate(data_loader):
                mri = mri.to(self.device)
                pet = pet.to(self.device)
                label = label.to(self.device)
                adas = adas.to(self.device)
                pred, mri_pred, pet_pred = self.net(mri, pet)
                loss = self.loss(label, adas, pred, mri_pred, pet_pred)

                prediction[i, :] = pred.cpu().squeeze().numpy()
                labels[i] = label.cpu().squeeze().numpy().argmax()
                benefit_array[i] = benefit.cpu().squeeze().numpy()
                loss_array[i] = loss.item()

        return prediction, labels, benefit_array, loss_array

    def compute_performance(self, pred, label, benefit_array):
        pred_label = pred.argmax(axis=1)
        true_boolean = pred_label == label
        accuracy = sum(true_boolean) / true_boolean.shape[0]
        benefit = benefit_array[true_boolean].sum()
        cm = confusion_matrix(label, pred_label)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        frp_ad = cm[0, 1] / (cm[0, 1] + cm[0, 0])
        one_hot_label = one_hot(label, self.num_classes)
        auc = roc_auc_score(one_hot_label, pred)

        return accuracy, sensitivity, specificity, frp_ad, auc, benefit

    def save_training_performance(self, epoch_performance):
        performance_path = os.path.join(
            root_path,
            'eval_result',
            self.model_name,
            'training_performance.csv'
        )
        for k, v in epoch_performance.items():
            if k not in self.training_performance:
                self.training_performance[k] = [v]
            else:
                self.training_performance[k].append(v)
        df = pd.DataFrame(self.training_performance)
        df.to_csv(performance_path, index=False)

    def train(self, num_epoch=100):
        print('开始训练')
        for epoch in range(1, num_epoch + 1):
            start_time = get_timestamp()

            # 训练一个epoch
            self.train_an_epoch()

            # 保存模型
            self.save_model(epoch)

            # 对训练集、数据集进行预测，并保存预测结果
            tr_pred, tr_lab, tr_ben_arr, tr_loss_arr = self.predict('train')
            te_pred, te_lab, te_ben_arr, te_loss_arr = self.predict('test')
            tr_pred_path = os.path.join(
                root_path,
                'predict_result',
                self.model_name,
                'train_{}.npy'.format(epoch)
            )
            te_pred_path = os.path.join(
                root_path,
                'predict_result',
                self.model_name,
                'test_{}.npy'.format(epoch)
            )
            np.save(tr_pred_path, tr_pred)
            np.save(te_pred_path, te_pred)

            # 计算各种指标：Accuracy,AUC,Sensitivity,Specificity,Benefit
            tr_acc, tr_sen, tr_spe, tr_frp, tr_auc, tr_ben = \
                self.compute_performance(tr_pred, tr_lab, tr_ben_arr)
            tr_loss = np.sum(tr_loss_arr) / tr_loss_arr.shape[0]
            te_acc, te_sen, te_spe, te_frp, te_auc, te_ben = \
                self.compute_performance(te_pred, te_lab, te_ben_arr)
            te_loss = np.sum(te_loss_arr) / te_loss_arr.shape[0]

            # 调整学习率
            self.lr_scheduler.step()

            # 保存各种训练数据：训练用时，loss，各种指标
            epoch_time = get_timestamp() - start_time
            epoch_performance = {
                'epoch': epoch,
                'time': epoch_time,
                'train_loss': tr_loss,
                'train_accuracy': tr_acc,
                'train_sensitivity': tr_sen,
                'train_specificity': tr_spe,
                'train_frp_ad': tr_frp,
                'train_auc': tr_auc,
                'train_benefit': tr_ben,
                'test_loss': te_loss,
                'test_accuracy': te_acc,
                'test_sensitivity': te_sen,
                'test_specificity': te_spe,
                'test_frp_ad': te_frp,
                'test_auc': te_auc,
                'test_benefit': te_ben
            }
            self.save_training_performance(epoch_performance)

            # 输出日志
            print(
                'Epoch {} -- {:.0f}s -- acc={:.4f} -- test_acc={:.4f} -- '
                'ben={:.4f} -- test_ben={:.4f}'.format(
                    epoch, epoch_time, tr_acc, te_acc, tr_ben, te_ben
                )
            )


if __name__ == '__main__':
    pass
