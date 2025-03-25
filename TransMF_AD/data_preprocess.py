import os
import numpy as np
import pandas as pd
from collections import Counter, deque
from config import root_path, zoom_mri_path 


def clean_dataset():
    data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
    total_data = pd.read_csv(os.path.join(root_path, 'datasets/ADNIMERGE_without_bad_value.csv'))
    total_data = total_data[['RID', 'VISCODE', 'ADAS13']]
    data = pd.merge(data, total_data, how='inner', on=['RID', 'VISCODE'])

    for c in data.columns:
        data = data[data[c].notna()]
    adas13 = data['ADAS13'].values
    data = data[(adas13 >= 0) & (adas13 <= 70)]
    data = data[data['DX'] != 'MCI']
    value_mapping = {'CN': 0, 'Dementia': 1}
    data['COG'] = [value_mapping[v] for v in data['DX'].values]
    data = data.drop(columns='DX')

    # 计算每个样本的benefit值
    data = data.sort_values(by=['RID', 'VISCODE'])
    rid = data['RID'].values
    adas = data['ADAS13'].values
    cog = data['COG'].values
    benefit_array = np.zeros(data.shape[0], dtype='float32')
    rid_mapping = {}
    for i in range(data.shape[0]):
        if rid[i] in rid_mapping:
            p = rid_mapping[rid[i]]
            benefit_array[p] = int(cog[p] == 1) * max(adas[i] - adas[p], 0)
        rid_mapping[rid[i]] = i
    for i in rid_mapping.values():
        benefit_array[i] = np.nan
    data['benefit'] = benefit_array

    data['ADAS13'] = (data['ADAS13'].values / 70).astype('float32')

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
    diff_value = allo_mat[-1, -1] - sum((~np.isnan(benefit)) & (y == 1))
    if diff_value <= 0:
        required_boolean = (~np.isnan(benefit)) & (y == 1)
        test_ad_index = np.where(required_boolean)[0][:allo_mat[-1, -1]]
        allo_boolean[-1, test_ad_index] = True
        # 校验
        assert allo_boolean[-1, :].sum() == allo_mat[-1, -1]
        assert (y[allo_boolean[-1, :]] != 1).sum() == 0
        assert np.isnan(benefit[allo_boolean[-1, :]]).sum() == 0
    else:
        allo_boolean[-1, (~np.isnan(benefit)) & (y == 1)] = True
        required_boolean = np.isnan(benefit) & (y == 1)
        diff_index = np.where(required_boolean)[0][:diff_value]
        allo_boolean[-1, diff_index] = True
        # 校验
        assert allo_boolean[-1, :].sum() == allo_mat[-1, -1]
        assert (y[allo_boolean[-1, :]] != 1).sum() == 0
        nan_counter = Counter(np.isnan(benefit[allo_boolean[-1, :]]))
        assert nan_counter[True] == diff_value
        assert nan_counter[False] == sum((~np.isnan(benefit)) & (y == 1))

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
        b[np.isnan(b) | (s['COG'] != 1)] = 0
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
        result[i].to_csv(fp, index=False)

    return result


def generate_test_dataset():
    """
           CN    MCI   AD    Sum
    train   1     1     1     3
    test    1     1     1     3
    Sum     2     2     2     6
    """
    # 载入、清除数据
    data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
    data = clean_dataset(data)

    # 打乱样本次序
    index = np.arange(data.shape[0], dtype='int')
    np.random.shuffle(index)
    data = data.iloc[index, :]

    # 生成测试数据
    benefit = data['benefit'].values
    y = data['COG'].values
    include_index = []
    allo_arr = [
        np.where(y == 0)[0][:2],
        np.where(y == 1)[0][:2],
        np.where((~np.isnan(benefit)) & (y == 2) & (benefit != 0))[0][:2]
    ]
    train_index = []
    test_index = []
    for i in range(3):
        train_index.append(allo_arr[i][0])
        test_index.append(allo_arr[i][1])

    # 分割数据集
    train_set = data.iloc[train_index, :]
    test_set = data.iloc[test_index, :]

    # 处理benefit
    for s in [train_set, test_set]:
        b = s['benefit'].values
        b[np.isnan(b) | (s['COG'] != 2)] = 0
        s.loc[:, 'benefit'] = b / b.sum()

    # 导出
    train_set.to_csv(
        os.path.join(root_path, 'datasets/_train.csv'),
        index=False
    )
    test_set.to_csv(
        os.path.join(root_path, 'datasets/_test.csv'),
        index=False
    )


def generate_small_dataset():
    data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
    data = clean_dataset(data)

    # 打乱样本次序
    index = np.arange(data.shape[0], dtype='int')
    np.random.shuffle(index)
    data = data.iloc[index, :]

    benefit = data['benefit'].values
    y = data['COG'].values
    num_sample = y.shape[0]
    train_boolean = np.zeros(num_sample, dtype='bool')
    test_boolean = np.zeros(num_sample, dtype='bool')

    condition = (~np.isnan(benefit)) & (y == 2) & (benefit != 0)
    condition_index = np.where(condition)[0]
    train_boolean[condition_index[:50]] = True
    test_boolean[condition_index[50:100]] = True
    for i in range(2):
        index = np.where(y == i)[0]
        train_boolean[index[:50]] = True
        test_boolean[index[50:100]] = True
    train_set = data[train_boolean]
    test_set = data[test_boolean]
    
    allo_arr = np.array([
        [50, 50, 50],
        [50, 50, 50]
    ], dtype='int')
    for r, s in enumerate([train_set, test_set]):
        s_y = s['COG'].values
        s_b = s['benefit'].values
        assert np.sum((s_y == 2) & np.isnan(s_b)) == 0
        assert np.sum((s_y == 2) & (s_b == 0)) == 0
        for c, count in Counter(s_y).items():
            assert count == allo_arr[r, c]

    for s in [train_set, test_set]:
        b = s['benefit'].values
        b[np.isnan(b) | (s['COG'].values != 2)] = 0
        s.loc[:, 'benefit'] = b / b.sum()

    train_set.to_csv(
        os.path.join(root_path, 'datasets/small_train.csv'),
        index=False
    )
    test_set.to_csv(
        os.path.join(root_path, 'datasets/small_test.csv'),
        index=False
    )


if __name__ == '__main__':
    data = clean_dataset()
    split_dataset(data, ratio=(8, 2), valid=False)
