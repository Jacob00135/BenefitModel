import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter, deque
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from config import root_path, mri_3d_path, mri_2d_path


def clean_dataset(data):
    data = data[['RID', 'VISCODE', 'filename', 'COG', 'ADAS13', 'benefit']]

    notna_required_columns = ['RID', 'VISCODE', 'filename', 'COG', 'ADAS13']
    for c in notna_required_columns:
        data = data[data[c].notna().values]

    data = data[data['COG'] != 1]
    cog_mapping = {0: 0, 2: 1}
    data['COG'] = [cog_mapping[v] for v in data['COG'].values]

    adas13 = data['ADAS13'].values
    data = data[(adas13 >=0) & (adas13 <= 70)]
    data['ADAS13'] = data['ADAS13'].values / 70

    return data


def get_dynamic_image(frames, normalized=True):
    """ Adapted from https://github.com/tcvrick/Python-Dynamic-Images-for-Action-Recognition"""
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    
    def _get_channel_frames(iter_frames, num_channels):
        """ Takes a list of frames and returns a list of frame lists split by channel. """
        frames = [[] for channel in range(num_channels)]

        for frame in iter_frames:
            for channel_frames, channel in zip(frames, cv2.split(frame)):
                channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
        for i in range(len(frames)):
            frames[i] = np.array(frames[i])
        return frames


    def _compute_dynamic_image(frames):
        """ Adapted from https://github.com/hbilen/dynamic-image-nets """
        num_frames, h, w, depth = frames.shape

        # Compute the coefficients for the frames.
        coefficients = np.zeros(num_frames)
        for n in range(num_frames):
            cumulative_indices = np.array(range(n, num_frames)) + 1
            coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)

        # Multiply by the frames by the coefficients and sum the result.
        x1 = np.expand_dims(frames, axis=0)
        x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
        result = x1 * x2
        return np.sum(result[0], axis=0).squeeze()

    num_channels = frames[0].shape[2]
    #print(num_channels)
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image


def transform_2d_mri(filenames):
    t = transforms.Resize((224, 224))
    for fn in tqdm(filenames):
        fp = os.path.join(mri_3d_path, fn)
        img = np.load(fp)
        img = np.expand_dims(img, -1)
        img = get_dynamic_image(img)
        img = Image.fromarray(img, 'L')
        img = t(img)
        img = np.array(img)
        img = np.expand_dims(img, 0)
        img = np.concatenate([img, img, img], 0)
        save_path = os.path.join(mri_2d_path, fn)
        np.save(save_path, img)


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
    # benefit[np.where(~np.isnan(benefit) & (y == 1))[0][:700]] = np.nan
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


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(root_path, 'datasets/data.csv'))
    data = clean_dataset(data)
    train_set, test_set = split_dataset(data, ratio=(8, 2), valid=False)

    """
    train_set = pd.read_csv(os.path.join(root_path, 'datasets/train.csv'))
    test_set = pd.read_csv(os.path.join(root_path, 'datasets/test.csv'))
    filenames = pd.concat((train_set, test_set))['filename'].values
    transform_2d_mri(filenames)
    """
