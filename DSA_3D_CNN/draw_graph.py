import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib.font_manager import FontProperties
from config import root_path

font_path = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
font = FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [str(font.get_name())]
plt.rcParams['axes.unicode_minus'] = False
"""
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
"""


def filtrate_dense_point(x, y, x_threshold=0.01, y_threshold=0.01, include_best_point=False):
    if x.shape[0] != y.shape[0]:
        raise ValueError('x和y长度不一致！')

    length = x.shape[0]
    filtrate_boolean = np.ones(length, dtype='bool')
    for i in range(length - 2):
        for j in range(i + 1, length):
            if abs(x[j] - x[i]) <= x_threshold and abs(y[j] - y[i]) <= y_threshold:
                filtrate_boolean[j] = False

    if not include_best_point:
        filtrate_boolean[x.argmax()] = True
        filtrate_boolean[y.argmax()] = True

    x = x[filtrate_boolean]
    y = y[filtrate_boolean]

    return x, y


def check_performance(model_name):
    path = os.path.join(
        root_path, 'eval_result', model_name, 'training_performance.csv'
    )
    data = pd.read_csv(path)
    check_columns = [
        'train_accuracy',
        'train_sensitivity',
        'train_specificity',
        'train_frp_cn_ad',
        'train_frp_mci_ad',
        'train_auc',
        'train_benefit',
        'test_accuracy',
        'test_sensitivity',
        'test_specificity',
        'test_frp_cn_ad',
        'test_frp_mci_ad',
        'test_auc',
        'test_benefit'
    ]
    invalid_columns = []
    for c in check_columns:
        var = data[c].values
        if (var < -1e-4).sum() + (var - 1 > 1e-4).sum() > 0:
            invalid_columns.append(c)
    if invalid_columns:
        raise ValueError('数据不合理：{}'.format(invalid_columns))


class MyPlot(object):

    def __init__(self, figsize=(8, 6), dpi=200):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)

    def plot(self, y, label, check=False):
        if check and ((y < -1e-4).sum() + (y - 1 > 1e-4).sum() > 0):
            raise ValueError('数据不合理')

        plt.plot(
            range(len(y)),
            y,
            marker='o',
            markersize=4,
            markeredgecolor='#555555',
            markeredgewidth=0.5,
            zorder=2,
            label=label
        )

    def config(self, title, ylabel, xlim=100):
        plt.title(title)
        plt.xlim(0, xlim)
        plt.ylim(0, 1)
        xticks = [-10]
        for i in range(xlim // 10 + 2):
            xticks.append(10 * i)
        plt.xticks(xticks)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        #plt.legend(loc='upper left')
        plt.grid(True, c='#eeeeee', ls='--', lw=0.5, zorder=0)

    def set_size_inches(self, t):
        self.fig.set_size_inches(*t)

    def adjust_padding(self, padding):
        plt.subplots_adjust(**padding)

    def show(self):
        plt.show()

    def save(self, filename):
        save_path = os.path.join(root_path, 'eval_plot', filename)
        plt.savefig(save_path)

    def close(self):
        plt.close()


def plot4(model_name_list):
    base_model_name = 'DSA_3D_CNN'
    index_mapping = {
        'Accuracy': 'test_accuracy',
        'Benefit': 'test_benefit',
        '$FPR_{AD}$': 'test_frp_ad',
        '总收益': 'total_index'
    }
    label_values = ['CrossEntropy', '$L_{ADAS}$', "$L^{'}_{FPR}$", '$L_{FPR}$']
    model_name_mapping = {}
    for k, v in zip(model_name_list, label_values):
        model_name_mapping[k] = v
    pic = MyPlot(figsize=(12, 9), dpi=400)
    for k, (index_name, c) in enumerate(index_mapping.items()):
        plt.subplot(2, 2, k + 1)
        for model_name, label in model_name_mapping.items():
            data_path = os.path.join(root_path, 'eval_result', model_name,
                                     'training_performance.csv')
            data = pd.read_csv(data_path)
            data['total_index'] = (data['test_accuracy'].values
                                   + data['test_benefit'].values
                                   + (1 - data['test_frp_ad'].values)) / 3
            y = data[c].values
            pic.plot(y, label, True)
        pic.config('{} {}'.format(base_model_name, index_name), index_name, xlim=100)
    plt.tight_layout()
    pic.save('{}.png'.format(base_model_name))
    pic.close()


def plot(model_name, xlim=100):
    pic = MyPlot()
    data = pd.read_csv(os.path.join(root_path, 'eval_result', model_name, 'training_performance.csv'))
    data['total_index'] = (data['test_accuracy'].values + data['test_benefit'].values + (1 - data['test_frp_ad'].values)) / 3
    index_list = ['test_accuracy', 'test_benefit', 'test_frp_ad', 'total_index']
    for index in index_list:
        y = data[index].values
        pic.plot(y=y, label=index, check=True)
    pic.config(title=model_name, ylabel='performance', xlim=xlim)
    # pic.show()
    pic.save('{}.png'.format(model_name))
    pic.close()


if __name__ == '__main__':
    plot4()
    # plot(model_name='src', xlim=300)
