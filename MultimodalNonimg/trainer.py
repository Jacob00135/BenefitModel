import os
import random
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp
from sklearn.metrics import roc_auc_score, confusion_matrix
from config import root_path
from data_preprocess import get_feature_columns
from models import MLP


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def one_hot(arr, num_classes):
    result = np.zeros((arr.shape[0], num_classes), dtype=arr.dtype)
    for i in range(result.shape[0]):
        result[i, arr[i]] = 1
    
    return result


class NonimgTrainer(object):

    def __init__(self, model_name, device='cpu', init_lr=0.001):
        super(NonimgTrainer, self).__init__()
        setup_seed(1124)

        # 初始化
        self.model_name = model_name
        self.device = device
        self.init_lr = init_lr
        self.num_classes = 3
        self.training_performance = {}
        generate_dir_list = [
            os.path.join(root_path, 'checkpoints', self.model_name),
            os.path.join(root_path, 'eval_result', self.model_name),
            os.path.join(root_path, 'predict_result', self.model_name)
        ]
        for p in generate_dir_list:
            if not os.path.exists(p):
                os.mkdir(p)

        # 载入数据集
        train_set_path = os.path.join(root_path, 'datasets/train.csv')
        test_set_path = os.path.join(root_path, 'datasets/test.csv')
        train_set = pd.read_csv(train_set_path)
        test_set = pd.read_csv(test_set_path)
        self.feature_columns = list(get_feature_columns(train_set.columns))
        self.x_train = train_set[self.feature_columns].to_numpy().astype('float32')
        self.y_train = one_hot(train_set['COG'].values, self.num_classes)
        self.x_test = test_set[self.feature_columns].to_numpy().astype('float32')
        self.y_test = one_hot(test_set['COG'].values, self.num_classes)
        self.adas_train = train_set['ADAS13'].values
        self.adas_test = test_set['ADAS13'].values
        self.benefit_train = train_set['benefit'].values
        self.benefit_test = test_set['benefit'].values
        self.data_loader_mapping = {
            'train': (self.x_train, self.y_train, self.adas_train, self.benefit_train),
            'test': (self.x_test, self.y_test, self.adas_test, self.benefit_test)
        }
        print('载入数据集成功')

        # 载入模型
        self.mlp = MLP(self.x_train.shape[1]).to(self.device)
        self.optim = torch.optim.Adam(self.mlp.parameters(), lr=self.init_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optim,
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
    def loss(pred, label, adas):
        return NonimgTrainer.cross_entropy_loss(pred, label)

    def train_an_epoch(self):
        self.mlp.train(True)
        self.optim.zero_grad()
        inputs = torch.tensor(self.x_train).to(self.device)
        labels = torch.tensor(self.y_train).to(self.device)
        preds = self.mlp(inputs)
        adas = torch.tensor(self.adas_train).to(self.device)
        loss = self.loss(preds, labels, adas)
        loss.backward()
        self.optim.step()

    def save_model(self, index):
        mlp_path = os.path.join(
            root_path,
            'checkpoints',
            self.model_name,
            '{}.pth'.format(index)
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
        x, labels, adas, benefit_array = self.data_loader_mapping[stage]

        # 预测
        self.mlp.eval()
        with torch.no_grad():
            prediction = self.mlp(torch.tensor(x).to(self.device))
            loss = self.loss(
                pred=prediction,
                label=torch.tensor(labels).to(self.device),
                adas=torch.tensor(adas).to(self.device)
            )
            prediction = prediction.cpu().squeeze().numpy()

        return prediction, labels, benefit_array, loss

    @staticmethod
    def compute_performance(pred, label, benefit_array):
        label = label.argmax(axis=1)
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
            tr_pred, tr_lab, tr_ben_arr, tr_loss = self.predict('train')
            te_pred, te_lab, te_ben_arr, te_loss = self.predict('test')
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
            te_acc, te_sen, te_spe, te_frp, te_auc, te_ben = \
                self.compute_performance(te_pred, te_lab, te_ben_arr)

            # 调整学习率
            # self.lr_scheduler.step()

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
