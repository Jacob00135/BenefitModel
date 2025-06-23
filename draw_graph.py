import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root_path = os.path.dirname(os.path.realpath(__file__))

font_path = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
font = FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [str(font.get_name())]
plt.rcParams['axes.unicode_minus'] = False


class MyPlot(object):

    def __init__(self, figsize=(8, 6), dpi=200):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)

        self.num_epoch = None

    def plot(self, y, label, check=False):
        if check and ((y < -1e-4).sum() + (y - 1 > 1e-4).sum() > 0):
            raise ValueError('数据不合理')

        self.num_epoch = len(y)
        x = range(1, self.num_epoch + 1)
        plt.plot(
            x,
            y,
            marker='o',
            markersize=4,
            markeredgecolor='#555555',
            markeredgewidth=0.5,
            zorder=2,
            label=label
        )

    def config(self, ylabel):
        plt.xlim(0, self.num_epoch)
        plt.ylim(0, 1)
        xticks = [-10]
        xticks.extend(range(0, self.num_epoch, 10))
        xticks.append(self.num_epoch)
        xticks.append(self.num_epoch + 10)
        plt.xticks(xticks)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, c='#eeeeee', ls='--', lw=0.5, zorder=0)

    def set_title(self, title, y):
        plt.title(title, loc='center', y=y, fontsize='x-large')

    def set_size_inches(self, t):
        self.fig.set_size_inches(*t)

    def adjust_padding(self, padding):
        plt.subplots_adjust(**padding)


if __name__ == '__main__':
    index_mapping = {
        'accuracy': {'ylabel': 'Accuracy', 'column': 'test_accuracy'},
        'benefit': {'ylabel': 'Benefit', 'column': 'test_benefit'},
        'fpr_ad': {'ylabel': '$FPR_{AD}$', 'column': 'test_frp_ad'},
        'total_benefit': {'ylabel': '总收益', 'column': 'total_benefit'}
    }
    legend_list = ['CrossEntropy', '$L_{ADAS}$', "$L^{'}_{FPR}$", '$L_{FPR}$']
    model_name_mapping = {
        'MultimodalMRI': {
            'sub_name': ('src', 'adas_1', 'frp', 'frp_1.125'),
            'title': 'Multimodal_MRI'
        },
        'MultimodalNonimg': {
            'sub_name': ('src', 'adas_2', 'frp', 'frp_2.5'),
            'title': 'Multimodal_Nonimg'
        },
        'MultimodalFusion': {
            'sub_name': ('src', 'adas_1', 'frp', 'frp_2.5'),
            'title': 'Multimodal_Fusion'
        },
        'DSA_3D_CNN': {
            'sub_name': ('src', 'adas_2', 'frp', 'frp_2'),
            'title': 'DSA_3D_CNN'
        },
        'TransMF_AD': {
            'sub_name': ('src', 'adas_2', 'frp', 'frp_2'),
            'title': 'TransMF_AD'
        },
        'ARP_2D_CNN': {
            'sub_name': ('src', 'adas_2', 'frp', 'frp_2'),
            'title': 'ARP_2D_CNN'
        }
    }
    for i, (index, config) in enumerate(index_mapping.items()):
        pic = MyPlot(figsize=(12, 13.5), dpi=400)
        for j, (model_name, item) in enumerate(model_name_mapping.items()):
            plt.subplot(3, 2, j + 1)
            for k, sub_name in enumerate(item['sub_name']):
                data_path = os.path.join(root_path, model_name, 'eval_result', sub_name, 'training_performance.csv')
                data = pd.read_csv(data_path)
                if config['column'] == 'total_benefit':
                    y = (data['test_accuracy'].values + data['test_benefit'].values + (1 - data['test_frp_ad'].values)) / 3
                else:
                    y = data[config['column']]
                pic.plot(y, legend_list[k], True)
            pic.config(ylabel=config['ylabel'])
            pic.set_title(
                title='({}){}模型的不同方案{}对比'.format(chr(ord('a') + j), item['title'], config['ylabel']),
                y=-0.23
            )
        plt.tight_layout()
        pic.adjust_padding({'hspace': 0.3})
        plt.savefig(os.path.join(root_path, 'global_eval_plot/图4.{}模型训练的{}指标趋势图（第二版）.png'.format(
            i + 1, config['ylabel']
        )))
        plt.close()

