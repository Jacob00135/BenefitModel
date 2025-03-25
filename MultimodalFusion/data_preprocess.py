import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import Counter, deque
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader
from config import root_path, zoom_mri_path
from models import _CNN_Bone_Zoom, MRIMLP
from dataloader import MRI3DDataset

multimodal_mri_root_path = os.path.join(root_path, '../MultimodalMRI')

"""
数据预处理的步骤：
1. 删除刚需特征（如COG）缺失的行
2. 删除缺失值比例过高的列（若超过50%直接删除，若位于[30%,50%]，考虑特征重要性）
3. 删除重复值比例过高的列（若重复率最高的值的重复率超过80%，则删除）
4. 删除全部都是重复值的行
- 4. 删除方差太小的列，因为方差太小意味着该列特征对于因变量来说没有区分度
- 5. 分别对每一个特征与因变量进行相关性检验，相关性过低的特征将被删除
- 5. 使用3sigma原则找出每个特征的异常条目，若检查发现确实是异常条目，则删除该条目
6. 填充缺失值（优先考虑随机森林方法，再考虑KNN方法）
7. 对有明确理论范围的数值型特征（如ADAS评分范围是明确的[0,70]）进行归一化
8. 对类别型特征进行独热编码处理
- 9. 使用随机森林等算法进行特征选择
10. 划分数据集为训练集、测试集，考虑类别不均衡问题，如果：
    样本容量最大的类别 : 样本容量最小的类别 > 2，可以视为类别不均衡
11. 对训练集、测试集的没有明确理论范围的数值型特征进行标准化，
    注意！标准化时使用的均值和方差要统一，都使用训练集的，否则造成数据泄露！
"""

all_columns = set([
    'RID', 'VISCODE', 'filename', 'ADAS13', 'COG', 'age', 'gender', 'benefit',
    'education', 'trailA', 'trailB', 'boston', 'digitB', 'digitBL', 'digitF',
    'digitFL', 'animal', 'gds', 'lm_imm', 'lm_del', 'mmse', 'npiq_DEL',
    'npiq_HALL', 'npiq_AGIT', 'npiq_DEPD', 'npiq_ANX', 'npiq_ELAT',
    'npiq_APA', 'npiq_DISN', 'npiq_IRR', 'npiq_MOT', 'npiq_NITE', 'npiq_APP',
    'faq_BILLS', 'faq_TAXES', 'faq_SHOPPING', 'faq_GAMES', 'faq_STOVE',
    'faq_MEALPREP', 'faq_EVENTS', 'faq_PAYATTN', 'faq_REMDATES', 'faq_TRAVEL',
    'his_NACCFAM', 'his_CVHATT', 'his_CVAFIB', 'his_CVANGIO', 'his_CVBYPASS',
    'his_CVPACE', 'his_CVCHF', 'his_CVOTHR', 'his_CBSTROKE', 'his_CBTIA',
    'his_SEIZURES', 'his_TBI', 'his_HYPERTEN', 'his_HYPERCHO', 'his_DIABETES',
    'his_B12DEF', 'his_THYROID', 'his_INCONTU', 'his_INCONTF', 'his_DEP2YRS',
    'his_DEPOTHR', 'his_PSYCDIS', 'his_ALCOHOL', 'his_TOBAC100', 'his_SMOKYRS',
    'his_PACKSPER', 'his_ABUSOTHR'
])
required_columns = set(['RID', 'VISCODE', 'filename', 'ADAS13', 'COG',
                        'benefit'])


def get_feature_columns(columns):
    return filter(lambda c: c not in required_columns, columns)


def clean_dataset_1(data):

    # 筛选掉不需要的列
    data = data[list(all_columns)]

    # 1. 删除刚需特征缺失的行
    filter_boolean = np.zeros(data.shape[0], dtype='bool')
    for c in required_columns:
        if c != 'benefit':
            filter_boolean = filter_boolean | (data[c].isna().values)
    data = data[~filter_boolean]
    data = data[(data['ADAS13'].values >= 0) & (data['ADAS13'].values <= 70)]

    # 2. 删除全部都是重复值的行
    exclude_column = ['RID', 'VISCODE', 'filename', 'ADAS13', 'COG']
    fc = [c for c in data.columns if c not in exclude_column]
    data = data[(data[fc].notna().sum(axis=1) > 0).values]

    # 3. 删除缺失值比例过高的列
    nan_thre1 = 0.5
    nan_thre2 = 0.3
    delete_mapping = {}
    for c in get_feature_columns(data.columns):
        nan_ratio = data[c].isna().sum() / data.shape[0]
        if nan_ratio > nan_thre2:
            delete_mapping[c] = nan_ratio
    delete_columns = [(k, v) for k, v in delete_mapping.items()]
    delete_columns = sorted(delete_columns, key=lambda t: t[1], reverse=True)
    for k, v in delete_columns:
        print('缺失：列{}比例{}'.format(k, v))
    data = data.drop(columns=delete_mapping.keys())

    # 4. 删除重复值比例过高的列
    print()
    repeat_ratio = 0.8
    delete_columns = []
    for c in get_feature_columns(data.columns):
        var = data[c].values[data[c].notna().values]
        counter = [(k, v) for k, v in Counter(var).items()]
        counter = sorted(counter, key=lambda t: t[1], reverse=True)
        value, count = counter[0]
        ratio = count / var.shape[0]
        if ratio > repeat_ratio:
            print('重复：列{}值{}比例{}'.format(c, value, ratio))
            delete_columns.append(c)
    data = data.drop(columns=delete_columns)

    return data


def split_dataset(data, ratio, valid=True):
    # 计算ratio
    ratio = list(ratio)
    ratio_sum = sum(ratio)
    for i in range(1, len(ratio)):
        ratio[i] = round(ratio[i] / ratio_sum, 1)
    ratio[0] = 1 - sum(ratio[1:])

    # 打乱样本次序
    index = np.arange(data.shape[0], dtype='int')
    np.random.shuffle(index)
    data = data.iloc[index, :]

    """
    以下为分割数据集的代码
    在分割数据集时，要考虑：
    1. 测试集的类别为AD的样本，benefit尽可能地不为nan；其他的可以随意
    2. train_cn:valid_cn:test_cn ≈
       train_mci:valid_mci:test_mci ≈
       train_ad:valid_ad:test_ad ≈ ratio

    测试分配(5:2:3)：
           CN    MCI   AD    Sum
    train  904  1421  735   3060
    valid  362  568   294   1224
    test   543  852   441   1836
    Sum   1809  2841  1470  6120

    实际分配(8:2)：
           CN    MCI   AD    Sum
    train 1447  2273  1176  4896
    test   362  568   294   1224
    Sum   1809  2841  1470  6120
    """

    # 生成分配矩阵
    y = data['COG'].values
    classes_counter = Counter(y)
    num_classes = len(classes_counter)
    if valid:
        allo_mat = np.zeros((3, num_classes), dtype='int')
    else:
        allo_mat = np.zeros((2, num_classes), dtype='int')
    for r in range(1, allo_mat.shape[0]):
        for c in range(num_classes):
            allo_mat[r, c] = round(classes_counter[c] * ratio[r])
    for c in range(num_classes):
        allo_mat[0, c] = classes_counter[c] - sum(allo_mat[1:, c])

    # 分配test_ad
    benefit = data['benefit'].values

    # region 测试用代码
    # benefit[np.where(~np.isnan(benefit) & (y == 2))[0][:700]] = np.nan
    # endregion

    allo_boolean = np.zeros(
        (allo_mat.shape[0], benefit.shape[0]),
        dtype='bool'
    )
    diff_value = allo_mat[-1, -1] - sum((~np.isnan(benefit)) & (y == 2))
    if diff_value <= 0:
        required_boolean = (~np.isnan(benefit)) & (y == 2)
        test_ad_index = np.where(required_boolean)[0][:allo_mat[-1, -1]]
        allo_boolean[-1, test_ad_index] = True
        # 校验
        assert allo_boolean[-1, :].sum() == allo_mat[-1, -1]
        assert (y[allo_boolean[-1, :]] != 2).sum() == 0
        assert np.isnan(benefit[allo_boolean[-1, :]]).sum() == 0
    else:
        allo_boolean[-1, (~np.isnan(benefit)) & (y == 2)] = True
        required_boolean = np.isnan(benefit) & (y == 2)
        diff_index = np.where(required_boolean)[0][:diff_value]
        allo_boolean[-1, diff_index] = True
        # 校验
        assert allo_boolean[-1, :].sum() == allo_mat[-1, -1]
        assert (y[allo_boolean[-1, :]] != 2).sum() == 0
        nan_counter = Counter(np.isnan(benefit[allo_boolean[-1, :]]))
        assert nan_counter[True] == diff_value
        assert nan_counter[False] == sum((~np.isnan(benefit)) & (y == 2))

    # 分配所有样本
    allo_queue = {}
    for c in range(allo_mat.shape[1]):
        queue = deque()
        for r in range(allo_mat.shape[0]):
            for i in range(allo_mat[r, c]):
                queue.append(r)
        allo_queue[c] = queue
    exclude_index = set(np.where(allo_boolean[-1, :])[0])
    for i, v in enumerate(y):
        if i in exclude_index:
            continue
        allo_boolean[allo_queue[v].popleft(), i] = True

    # 提取样本
    result = []
    for r in range(allo_boolean.shape[0]):
        result.append(data[allo_boolean[r, :]])

    # 处理Benefit
    for s in result:
        b = s['benefit'].values
        b[np.isnan(b) | (s['COG'] != 2)] = 0
        s.loc[:, 'benefit'] = b / b.sum()

    # 校验
    for r in range(allo_mat.shape[0]):
        counter = Counter(result[r]['COG'].values)
        for c in range(allo_mat.shape[1]):
            assert counter[c] == allo_mat[r, c]

    # 保存
    filenames = ['train.csv', 'test.csv']
    if valid:
        filenames.insert(1, 'valid.csv')
    for i, fn in enumerate(filenames):
        fp = os.path.join(root_path, 'datasets', fn)
        # result[i].to_csv(fp, index=False)

    return result


def clean_dataset_2(train_set, test_set):
    """
    数据预处理的步骤：
    1. 删除刚需特征（如COG）缺失的行
    2. 删除缺失值比例过高的列（若超过50%直接删除，若位于[30%,50%]，考虑特征重要性）
    3. 删除重复值比例过高的列（若重复率最高的值的重复率超过80%，则删除）
    - 4. 删除方差太小的列，因为方差太小意味着该列特征对于因变量来说没有区分度
    - 5. 分别对每一个特征与因变量进行相关性检验，相关性过低的特征将被删除
    - 5. 使用3sigma原则找出每个特征的异常条目，若检查发现确实是异常条目，则删除该条目
    6. 填充缺失值（优先考虑随机森林方法，再考虑KNN方法）
    7. 对有明确理论范围的数值型特征（如ADAS评分范围是明确的[0,70]）进行归一化
    8. 对类别型特征进行独热编码处理
    9. 使用随机森林等算法进行特征选择
    10. 划分数据集为训练集、测试集，考虑类别不均衡问题，如果：
        样本容量最大的类别 : 样本容量最小的类别 > 2，可以视为类别不均衡
    11. 对训练集、测试集的没有明确理论范围的数值型特征进行标准化，
        注意！标准化时使用的均值和方差要统一，都使用训练集的，否则造成数据泄露！
    """

    # 5. 填充缺失值：随机森林方法
    for c in get_feature_columns(train_set.columns):
        var = np.append(train_set[c].values, test_set[c].values)
        if np.isnan(var).sum() <= 0:
            continue

        # 训练随机森林模型
        notna_train_set = train_set[train_set[c].notna()]
        drop_columns = [c, 'RID', 'VISCODE', 'filename', 'COG', 'benefit']
        x_train = notna_train_set.drop(columns=drop_columns).to_numpy()
        y_train = notna_train_set[c].values
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        
        # 填充训练集缺失
        x = train_set[train_set[c].isna()].drop(columns=drop_columns).to_numpy()
        if x.shape[0] > 0:
            train_set.loc[train_set[c].isna(), c] = model.predict(x)

        # 填充测试集缺失
        x = test_set[test_set[c].isna()].drop(columns=drop_columns).to_numpy()
        if x.shape[0] > 0:
            test_set.loc[test_set[c].isna(), c] = model.predict(x)

    # 6. 对有明确理论范围的数值型特征（如ADAS评分范围是明确的[0,70]）进行归一化
    normalize_mapping = {'faq_TAXES': 5, 'faq_MEALPREP': 5, 'faq_EVENTS': 5,
                         'faq_TRAVEL': 5, 'faq_BILLS': 5, 'faq_PAYATTN' :5,
                         'faq_GAMES': 5, 'faq_REMDATES': 5, 'gds': 30,
                         'mmse': 30, 'boston': 30, 'animal': 60, 'age': 100,
                         'ADAS13': 70}
    for c, m in normalize_mapping.items():
        var = train_set[c].values
        train_set = train_set[(var >= 0) & (var <= m)]
        var = test_set[c].values
        test_set = test_set[(var >= 0) & (var <= m)]
    for c, m in normalize_mapping.items():
        train_set.loc[:, c] = train_set[c].values / m
        test_set.loc[:, c] = test_set[c].values / m

    # 保存
    # train_set.to_csv(os.path.join(root_path, 'datasets/train.csv'), index=False)
    # test_set.to_csv(os.path.join(root_path, 'datasets/test.csv'), index=False)

    return train_set, test_set


def add_mri_feature(dataframe, device='cpu'):
    # 寻找出最优的中间模型
    model_name = 'frp_1.125'
    training_performance = pd.read_csv(os.path.join(multimodal_mri_root_path, 'eval_result', model_name, 'training_performance.csv'))
    benefit = training_performance['test_benefit'].values
    accuracy = training_performance['test_accuracy'].values
    frp_ad = training_performance['test_frp_ad'].values
    total_index = (benefit + accuracy + (1 - frp_ad)) / 3
    training_performance['total_index'] = total_index
    training_performance = training_performance.sort_values(by='total_index', ascending=False)
    training_performance.index = range(training_performance.shape[0])
    best_epoch = training_performance.loc[0, 'epoch']
    backbone_path = os.path.join(multimodal_mri_root_path, 'checkpoints', model_name, 'backbone_{}.pth'.format(best_epoch))
    mlp_path = os.path.join(multimodal_mri_root_path, 'checkpoints', model_name, 'MLP_{}.pth'.format(best_epoch))

    # 载入最优中间模型
    backbone = _CNN_Bone_Zoom().to(device)
    mlp = MRIMLP(backbone.size).to(device)
    print('载入模型成功')

    # 使用最优中间模型预测
    dataset = MRI3DDataset(dataframe, mri_dir_path=zoom_mri_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    num_sample = len(dataset)
    num_classes = 3
    prediction = np.zeros((num_sample, num_classes), dtype='float32')
    backbone.eval()
    mlp.eval()
    with torch.no_grad():
        for i, (img, label, record) in enumerate(data_loader):
            prediction[i, :] = mlp(backbone(img.to(device))).cpu().squeeze().numpy()
    print('预测成功')

    # 特征融合
    for i, c in enumerate(['CN_prob', 'MCI_prob', 'AD_prob']):
        dataframe[c] = prediction[:, i]
    print('特征融合成功')

    return dataframe


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
    data = clean_dataset_1(data)
    train_set, test_set = split_dataset(data, ratio=(8, 2), valid=False)
    train_set, test_set = clean_dataset_2(train_set, test_set)
    data = add_mri_feature(pd.concat((train_set, test_set)), device='cuda:0')
    data.iloc[:train_set.shape[0], :].to_csv(os.path.join(root_path, 'datasets/train.csv'), index=False)
    data.iloc[train_set.shape[0]:, :].to_csv(os.path.join(root_path, 'datasets/test.csv'), index=False)
    print('预处理完毕')
