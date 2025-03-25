import os

root_path = os.path.dirname(__file__)
mri_3d_path = '/home/xjy/ADNI_MRI_npy'
mri_2d_path = os.path.join(root_path, 'datasets/2d_mri')

required_dir = ['checkpoints', 'eval_plot', 'eval_result', 'predict_result']
for dir_name in required_dir:
    path = os.path.join(root_path, dir_name)
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    print(root_path)
