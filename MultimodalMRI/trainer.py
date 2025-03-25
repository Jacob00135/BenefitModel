import os
import random
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from config import root_path, zoom_mri_path, mri_path
from models import MRI3DModel, _CNN_Bone, MLP, _CNN_Bone_Zoom
from dataloader import MRI3DDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MRITrainer(object):

    def __init__(self,
        model_name, device='cpu', batch_size=16, init_lr=0.001,
        small_dataset=False, zoom_dataset=False
    ):
        setup_seed(1124)

        # 初始化
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.num_classes = 3
        self.training_performance = {}
        self.small_dataset = small_dataset
        self.zoom_dataset = zoom_dataset
        if self.small_dataset:
            self.dataset_mapping = {
                'train': 'small_train.csv', 'test': 'small_test.csv'
            }
        else:
            self.dataset_mapping = {
                'train': 'train.csv', 'test': 'test.csv'
            }
        if self.zoom_dataset:
            self.mri_dir_path = zoom_mri_path
        else:
            self.mri_dir_path = mri_path
        generate_dir_list = [
            os.path.join(root_path, 'checkpoints', self.model_name),
            os.path.join(root_path, 'eval_result', self.model_name),
            os.path.join(root_path, 'predict_result', self.model_name)
        ]
        for p in generate_dir_list:
            if not os.path.exists(p):
                os.mkdir(p)

        # 载入数据集
        train_set_path = os.path.join(
            root_path,
            'datasets',
            self.dataset_mapping['train']
        )
        test_set_path = os.path.join(
            root_path,
            'datasets',
            self.dataset_mapping['test']
        )
        train_set = pd.read_csv(train_set_path)
        test_set = pd.read_csv(test_set_path)
        self.train_set = MRI3DDataset(train_set, mri_dir_path=self.mri_dir_path)
        self.test_set = MRI3DDataset(test_set, mri_dir_path=self.mri_dir_path)
        self.train_dataloader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        self.test_dataloader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        print('载入数据集成功')

        # 载入模型
        if zoom_dataset:
            self.backbone = _CNN_Bone_Zoom().to(self.device)
        else:
            self.backbone = _CNN_Bone.to(self.device)
        self.mlp = MLP(self.backbone.size).to(self.device)
        self.backbone_optim = Adam(
            self.backbone.parameters(), lr=self.init_lr, betas=(0.5, 0.999)
        )
        self.mlp_optim = Adam(self.mlp.parameters(), lr=self.init_lr)
        self.backbone_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.backbone_optim,
            gamma=0.98
        )
        self.mlp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.mlp_optim,
            gamma=0.98
        )
        print('载入模型成功')

    @staticmethod
    def cross_entropy_loss(pred, label):
        delta = 1e-10

        return torch.div(
            -1 * torch.sum(label * torch.log(pred + delta)),
            pred.shape[0]
        )

    @staticmethod
    def loss(pred, label, record):
        return MRITrainer.cross_entropy_loss(pred, label)

    def train_an_epoch(self):
        self.backbone.train(True)
        self.mlp.train(True)
        for img, label, record in self.train_dataloader:
            self.backbone_optim.zero_grad()
            self.mlp_optim.zero_grad()
            img = img.to(self.device)
            label = label.to(self.device)
            pred = self.mlp(self.backbone(img))
            loss = self.loss(pred, label, record)
            loss.backward()
            self.backbone_optim.step()
            self.mlp_optim.step()

    def save_model(self, index):
        backbone_path = os.path.join(
            root_path,
            'checkpoints',
            self.model_name,
            'backbone_{}.pth'.format(index)
        )
        torch.save(self.backbone.state_dict(), backbone_path)

        mlp_path = os.path.join(
            root_path,
            'checkpoints',
            self.model_name,
            'MLP_{}.pth'.format(index)
        )
        torch.save(self.mlp.state_dict(), mlp_path)

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
        path = os.path.join(root_path, 'datasets', self.dataset_mapping[stage])
        dataset = MRI3DDataset(
            pd.read_csv(path),
            mri_dir_path=self.mri_dir_path
        )
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
        self.backbone.eval()
        self.mlp.eval()
        with torch.no_grad():
            for i, (img, label, record) in enumerate(data_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                pred = self.mlp(self.backbone(img))
                loss = self.loss(pred, label, record)

                prediction[i, :] = pred.cpu().squeeze().numpy()
                labels[i] = label.cpu().squeeze().numpy().argmax()
                benefit_array[i] = record['benefit'].cpu().squeeze().numpy()
                loss_array[i] = loss.item()

        return prediction, labels, benefit_array, loss_array

    @staticmethod
    def compute_performance(pred, label, benefit_array):
        pred_label = pred.argmax(axis=1)
        true_boolean = pred_label == label
        accuracy = sum(true_boolean) / true_boolean.shape[0]
        benefit = benefit_array[true_boolean].sum()
        auc = roc_auc_score(
            label,
            pred,
            multi_class='ovr',
            average='weighted'
        )
        sensitivity = 0
        specificity = 0
        for i in range(3):
            class_label = (label == i).astype('int')
            class_pred_label = (pred_label == i).astype('int')
            cm = confusion_matrix(class_label, class_pred_label)
            sensitivity = sensitivity + cm[1, 1] / (cm[1, 1] + cm[1, 0])
            specificity = specificity + cm[0, 0] / (cm[0, 0] + cm[0, 1])
        sensitivity = sensitivity / 3
        specificity = specificity / 3

        three_cm = confusion_matrix(label, pred_label)
        frp_ad = (three_cm[0, 2] + three_cm[1, 2]) / three_cm[:2, :].sum()

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
            self.backbone_lr_scheduler.step()
            self.mlp_lr_scheduler.step()

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
