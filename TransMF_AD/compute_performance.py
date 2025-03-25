import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from config import root_path
from draw_graph import plot, plot4


def compute_frp_ad(model_name_list):
    for model_name in model_name_list:
        # 读取training_performance
        training_performance_path = os.path.join(
            root_path,
            'eval_result',
            model_name,
            'training_performance.csv'
        )
        delete_columns = [
            'train_frp_cn_ad', 'train_frp_mci_ad',
            'test_frp_cn_ad', 'test_frp_mci_ad'
        ]
        data = pd.read_csv(training_performance_path)
        data = data.drop(columns=[c for c in delete_columns if c in data.columns])

        # 读取y_true
        y_true_mapping = {
            'train': pd.read_csv(
                os.path.join(root_path, 'datasets/train.csv')
            )['COG'].values,
            'test': pd.read_csv(
                os.path.join(root_path, 'datasets/test.csv')
            )['COG'].values
        }

        # 读取y_pred并计算frp_ad
        index_mapping = {
            'train': np.zeros(data.shape[0], dtype='float32'),
            'test': np.zeros(data.shape[0], dtype='float32')
        }
        pred_path = os.path.join(root_path, 'predict_result', model_name)
        for fn in os.listdir(pred_path):
            dataset_name, epoch = fn.rsplit('.', 1)[0].rsplit('_', 1)
            y_pred = np.load(os.path.join(pred_path, fn)).argmax(axis=1)
            y_true = y_true_mapping[dataset_name]
            cm = confusion_matrix(y_true, y_pred)
            frp_ad = (cm[0, 2] + cm[1, 2]) / cm[:2, :].sum()
            index_mapping[dataset_name][int(epoch) - 1] = frp_ad
        data['train_frp_ad'] = index_mapping['train']
        data['test_frp_ad'] = index_mapping['test']

        # 保存
        data.to_csv(training_performance_path, index=False)


def print_info(model_name_list):
    index_name_mapping = OrderedDict({
        'test_accuracy': 'Accuracy',
        'test_benefit': 'Benefit',
        'test_frp_ad': 'FRP_AD',
        'total_index': '总收益'
    })
    result = {v:[] for v in index_name_mapping.values()}
    result['model'] = []
    for model_name in model_name_list:
        training_performance_path = os.path.join(
            root_path,
            'eval_result',
            model_name,
            'training_performance.csv'
        )
        training_performance = pd.read_csv(training_performance_path)
        benefit = training_performance['test_benefit'].values
        accuracy = training_performance['test_accuracy'].values
        frp_ad = training_performance['test_frp_ad'].values
        total_index = (benefit + accuracy + (1 - frp_ad)) / 3
        training_performance['total_index'] = total_index
        training_performance = training_performance.sort_values(
            by='total_index', ascending=False
        )
        best_index = training_performance.iloc[0, :]
        for k, v in index_name_mapping.items():
            result[v].append(round(best_index[k], 4))
        result['model'].append(model_name)
    result_df = pd.DataFrame(result)
    result_df = result_df.set_index('model')
    print(result_df)


if __name__ == '__main__':
    model_name_list = ['src', 'adas_2', 'frp', 'frp_2']
    print_info(model_name_list)
    plot4(model_name_list)
    for model_name in model_name_list:
        plot(model_name)
