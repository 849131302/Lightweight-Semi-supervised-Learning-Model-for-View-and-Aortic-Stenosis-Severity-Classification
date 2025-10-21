import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore",category=UserWarning)

from torch.nn import functional as F

def TMED2_View():
    # 读取包含 query_key、view_label、diagnosis_label 标签的 CSV 文件
    data = pd.read_csv(
        'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\DEV165\TMED2_fold0_labeledpart.csv')  # D:/cv/Dataset/TMED2/approved_users_only/labels_per_image.csv
    print(data)
    # 图像数据集路径
    #image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\view_and_diagnosis_labeled_set\labeled'
    #image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\view_and_diagnosis_labeled_set\\unlabeled'
    #image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\view_labeled_set\labeled'
    #image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\view_labeled_set\\unlabeled'
    image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\unlabeled_set\\unlabeled_set'
    # 创建 ImageNet类型数据集目录结构#
    output_dir = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV165(S0)'  # 存储数据集的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    # 遍历CSV文件中的每一行数据
    for index, row in data.iterrows():
        print(index)
        query_key = row['query_key']
        classifier_split = row['view_classifier_split']
        view_label = row['view_label']
        diagnosis_label = row['diagnosis_label']

        # 根据 query_key 从图像数据集中找到对应的图像文件
        image_file = os.path.join(image_data_path, f"{query_key}")  # 图像文件的路径

        # 检查图像文件是否存在
        if os.path.exists(image_file):
            if classifier_split == 'train':  # and (view_label=="PLAX" or view_label=="PSAX"):
                if view_label == 'PLAX':  # 创建对应的类别文件夹
                    class_dir = os.path.join(output_dir, 'train', str('PLAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'PSAX':
                    class_dir = os.path.join(output_dir, 'train', str('PSAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A2C':
                    class_dir = os.path.join(output_dir, 'train', str('A2C'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A4C':
                    class_dir = os.path.join(output_dir, 'train', str('A4C'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A4CorA2CorOther':
                    class_dir = os.path.join(output_dir, 'train', str('A4CorA2CorOther'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
            elif classifier_split == 'val':
                if view_label == 'PLAX':  # 创建对应的类别文件夹
                    class_dir = os.path.join(output_dir, 'val', str('PLAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'PSAX':
                    class_dir = os.path.join(output_dir, 'val', str('PSAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A2C':
                    class_dir = os.path.join(output_dir, 'val', str('A2C'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A4C':
                    class_dir = os.path.join(output_dir, 'val', str('A4C'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A4CorA2CorOther':
                    class_dir = os.path.join(output_dir, 'val', str('A4CorA2CorOther'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
            elif classifier_split == 'test':
                if view_label == 'PLAX':  # 创建对应的类别文件夹
                    class_dir = os.path.join(output_dir, 'test', str('PLAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'PSAX':
                    class_dir = os.path.join(output_dir, 'test', str('PSAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A2C':
                    class_dir = os.path.join(output_dir, 'test', str('A2C'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A4C':
                    class_dir = os.path.join(output_dir, 'test', str('A4C'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'A4CorA2CorOther':
                    class_dir = os.path.join(output_dir, 'test', str('A4CorA2CorOther'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std




def TMED1_PLAXPSAX():
    # 读取包含 query_key、view_label、diagnosis_label 标签的 CSV 文件
    data = pd.read_csv(
        '/home/CV/AS/TMED1/approved_users_only/TMED-156-52_fold0.csv')  # D:/cv/Dataset/TMED2/approved_users_only/labels_per_image.csv
    print(data)
    # 图像数据集路径
    image_data_path = '/home/CV/AS/TMED1/approved_users_only/partially_labeled'
    #image_data_path = '/home/CV/AS/TMED/approved_users_only/view_labeled_set/labeled'
    # 创建 ImageNet类型数据集目录结构
    output_dir = '/home/CV/AS/TMED1/TMED-156-52_fold0'  # 存储数据集的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    # 遍历CSV文件中的每一行数据
    for index, row in data.iterrows():
        print(index)
        query_key = row['query_key']
        classifier_split = row['split']
        view_label = row['view_label']
        diagnosis_label = row['diagnosis_label']

        # 根据 query_key 从图像数据集中找到对应的图像文件
        image_file = os.path.join(image_data_path, f"{query_key}")  # 图像文件的路径

        # 检查图像文件是否存在
        if os.path.exists(image_file):
            if classifier_split == 'train':
                if diagnosis_label == 'no_as':
                    class_dir = os.path.join(output_dir, 'train', str('no_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'mild/moderate_as':
                    class_dir = os.path.join(output_dir, 'train', str('mildmoderate_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'severe_as':
                    class_dir = os.path.join(output_dir, 'train', str('severe_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
            elif classifier_split == 'val' :
                if diagnosis_label == 'no_as':
                    class_dir = os.path.join(output_dir, 'val', str('no_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'mild/moderate_as':
                    class_dir = os.path.join(output_dir, 'val', str('mildmoderate_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'severe_as':
                    class_dir = os.path.join(output_dir, 'val', str('severe_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
            elif classifier_split == 'test' :
                if diagnosis_label == 'no_as':
                    class_dir = os.path.join(output_dir, 'test', str('no_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'mild/moderate_as':
                    class_dir = os.path.join(output_dir, 'test', str('mildmoderate_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'severe_as':
                    class_dir = os.path.join(output_dir, 'test', str('severe_as'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
def TMED1_View():
    # 读取包含 query_key、view_label、diagnosis_label 标签的 CSV 文件
    data = pd.read_csv(
        'D:\CV\LWM-SSL\TMED\TMED-1\TMED-156-52_fold2.csv')  # D:/cv/Dataset/TMED2/approved_users_only/labels_per_image.csv
    print(data)
    # 图像数据集路径
    #image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-1\labeled'
    #image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\view_and_diagnosis_labeled_set\\unlabeled'
    #image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-1\partially_labeled'
    image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-1\\unlabeled'
    # 创建 ImageNet类型数据集目录结构#
    output_dir = 'D:\CV\LWM-SSL\TMED\TMED-1\TMED-3VIEW\SPLIT2'  # 存储数据集的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    # 遍历CSV文件中的每一行数据
    for index, row in data.iterrows():
        print(index)
        query_key = row['query_key']
        classifier_split = row['split']
        view_label = row['view_label']
        diagnosis_label = row['diagnosis_label']

        # 根据 query_key 从图像数据集中找到对应的图像文件
        image_file = os.path.join(image_data_path, f"{query_key}")  # 图像文件的路径

        # 检查图像文件是否存在
        if os.path.exists(image_file):
            if classifier_split == 'train':  # and (view_label=="PLAX" or view_label=="PSAX"):
                if view_label == 'PLAX':  # 创建对应的类别文件夹
                    class_dir = os.path.join(output_dir, 'train', str('PLAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'PSAX AoV':
                    class_dir = os.path.join(output_dir, 'train', str('PSAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'Other':
                    class_dir = os.path.join(output_dir, 'train', str('Other'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
            elif classifier_split == 'val':
                if view_label == 'PLAX':  # 创建对应的类别文件夹
                    class_dir = os.path.join(output_dir, 'val', str('PLAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'PSAX AoV':
                    class_dir = os.path.join(output_dir, 'val', str('PSAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'Other':
                    class_dir = os.path.join(output_dir, 'val', str('Other'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
            elif classifier_split == 'test':
                if view_label == 'PLAX':  # 创建对应的类别文件夹
                    class_dir = os.path.join(output_dir, 'test', str('PLAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'PSAX AoV':
                    class_dir = os.path.join(output_dir, 'test', str('PSAX'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif view_label == 'Other':
                    class_dir = os.path.join(output_dir, 'test', str('Other'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))

def TMED1_AS():
    # 读取包含 query_key、view_label、diagnosis_label 标签的 CSV 文件
    data = pd.read_csv('/home/CV/AS/TMED1/approved_users_only/TMED-156-52_fold3.csv')  # D:/cv/Dataset/TMED2/approved_users_only/labels_per_image.csv
    print(data)
    # 图像数据集路径
    #image_data_path = '/home/CV/AS/TMED1/approved_users_only/partially_labeled'
    #image_data_path = '/home/CV/AS/TMED1/approved_users_only/labeled'
    image_data_path = '/home/CV/AS/TMED1/approved_users_only/unlabeled'
    # 创建 ImageNet类型数据集目录结构
    output_dir = '/home/CV/AS/TMED1/TMED-156-52_fold3'  # 存储数据集的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    # 遍历CSV文件中的每一行数据
    for index, row in data.iterrows():
        print(index)
        query_key = row['query_key']
        classifier_split = row['split']
        diagnosis_label = row['diagnosis_label']

        # 根据 query_key 从图像数据集中找到对应的图像文件
        image_file = os.path.join(image_data_path, f"{query_key}")  # 图像文件的路径

        # 检查图像文件是否存在
        if os.path.exists(image_file):
            # if classifier_split == 'train':
            #     if diagnosis_label == 'no_as':
            #         class_dir = os.path.join(output_dir, 'train', str('no_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            #     elif diagnosis_label == 'mild/moderate_as':
            #         class_dir = os.path.join(output_dir, 'train', str('mildmoderate_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            #     elif diagnosis_label == 'severe_as':
            #         class_dir = os.path.join(output_dir, 'train', str('severe_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            # elif classifier_split == 'val' :
            #     if diagnosis_label == 'no_as':
            #         class_dir = os.path.join(output_dir, 'val', str('no_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            #     elif diagnosis_label == 'mild/moderate_as':
            #         class_dir = os.path.join(output_dir, 'val', str('mildmoderate_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            #     elif diagnosis_label == 'severe_as':
            #         class_dir = os.path.join(output_dir, 'val', str('severe_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            # elif classifier_split == 'test' :
            #     if diagnosis_label == 'no_as':
            #         class_dir = os.path.join(output_dir, 'test', str('no_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            #     elif diagnosis_label == 'mild/moderate_as':
            #         class_dir = os.path.join(output_dir, 'test', str('mildmoderate_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            #     elif diagnosis_label == 'severe_as':
            #         class_dir = os.path.join(output_dir, 'test', str('severe_as'))
            #         os.makedirs(class_dir, exist_ok=True)
            #         shutil.copy(image_file, class_dir)
            #         print('image:{} done'.format(query_key))
            if classifier_split == 'Unlabeled':
                class_dir = os.path.join(output_dir, 'Unlabeled', str('no_as'))
                os.makedirs(class_dir, exist_ok=True)
                shutil.copy(image_file, class_dir)
                print('image:{} done'.format(query_key))




def TMED2_PLAXPSAX():
    # 读取包含 query_key、view_label、diagnosis_label 标签的 CSV 文件
    data = pd.read_csv(
        'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\DEV479\TMED2_fold0_labeledpart.csv')  # D:/cv/Dataset/TMED2/approved_users_only/labels_per_image.csv
    print(data)
    # 图像数据集路径
    image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\view_and_diagnosis_labeled_set\labeled'
    #image_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\approved_users_only\view_labeled_set\labeled'
    # 创建 ImageNet类型数据集目录结构
    output_dir = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-AS\TMED2-PLAXPSAX'  # 存储数据集的目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    # 遍历CSV文件中的每一行数据
    for index, row in data.iterrows():
        print(index)
        query_key = row['query_key']
        classifier_split = row['diagnosis_classifier_split']
        view_label = row['view_label']
        diagnosis_label = row['diagnosis_label']

        # 根据 query_key 从图像数据集中找到对应的图像文件
        image_file = os.path.join(image_data_path, f"{query_key}")  # 图像文件的路径

        # 检查图像文件是否存在
        if os.path.exists(image_file):
            if classifier_split == 'train' and (view_label=="PLAX" or view_label=="PSAX"):
                if diagnosis_label == 'no_AS':
                    class_dir = os.path.join(output_dir, 'train', str('no_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'mild_AS' or diagnosis_label == 'mildtomod_AS':
                    class_dir = os.path.join(output_dir, 'train', str('early_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'moderate_AS' or diagnosis_label == 'severe_AS':
                    class_dir = os.path.join(output_dir, 'train', str('significant_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))

            elif classifier_split == 'val' and (view_label=="PLAX" or view_label=="PSAX"):
                if diagnosis_label == 'no_AS':
                    class_dir = os.path.join(output_dir, 'val', str('no_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'mild_AS' or diagnosis_label == 'mildtomod_AS':
                    class_dir = os.path.join(output_dir, 'val', str('early_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'moderate_AS' or diagnosis_label == 'severe_AS':
                    class_dir = os.path.join(output_dir, 'val', str('significant_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
            elif classifier_split == 'test' and (view_label=="PLAX" or view_label=="PSAX"):
                if diagnosis_label == 'no_AS':
                    class_dir = os.path.join(output_dir, 'test', str('no_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'mild_AS' or diagnosis_label == 'mildtomod_AS':
                    class_dir = os.path.join(output_dir, 'test', str('early_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
                elif diagnosis_label == 'moderate_AS' or diagnosis_label == 'severe_AS':
                    class_dir = os.path.join(output_dir, 'test', str('significant_AS'))
                    os.makedirs(class_dir, exist_ok=True)
                    shutil.copy(image_file, class_dir)
                    print('image:{} done'.format(query_key))
def AS_Patient():
    # 源文件夹路径
    source_folder = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-AS\DEV479\\test'
    # 获取test所在的目录
    parent_folder = os.path.dirname(source_folder)
    # 目标文件夹路径
    target_folder = os.path.join(parent_folder, 'test_patient')
    # 遍历每个子文件夹
    for subfolder in ['early_AS', 'no_AS', 'significant_AS']:
        subfolder_path = os.path.join(source_folder, subfolder)
        # 遍历子文件夹中的所有图片文件
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.png'):
                # 从文件名中提取患者名（这里假设患者名是文件名的前缀）
                patient_name = filename.split('_')[0]
                # 创建以患者名命名的子文件夹，如果不存在
                patient_folder_path = os.path.join(target_folder, subfolder, patient_name)
                os.makedirs(patient_folder_path, exist_ok=True)
                # 移动图片到相应的子文件夹
                source_filepath = os.path.join(subfolder_path, filename)
                destination_filepath = os.path.join(patient_folder_path, filename)
                shutil.copy(source_filepath, destination_filepath)
    print("提取和复制完成。")

if __name__ == '__main__':
    #TMED2_PLAXPSAX()
    #TMED_PLAXPSAX()
    #TMED1_PLAXPSAX()
    #TMED1_AS()
    #AS_Patient()
    TMED1_View()


    '''labeled_data_path = '/home/CV/AS/TMED1/TMED-156-52_fold0/train'
    # unlabeled_data_path = '/home/CV/AS/TMED/unlabeled'  # unlabeled_set
    # val_data_path = '/home/CV/AS/TMED/TMED-VIEW-DEV56/val'
    # test_data_path = '/home/CV/AS/TMED/TMED-VIEW-DEV56/test'
    #
    # class TransformTwice:
    #     def __init__(self, weak_transform, strong_transform):
    #         self.weak_transform = weak_transform
    #         self.strong_transform = strong_transform
    #     def __call__(self, x):
    #         ulb_w = self.weak_transform(x)
    #         ulb_s = self.strong_transform(x)
    #         return ulb_w, ulb_s
    transform_labeledtrain = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.Resize(size=(112,112), interpolation=InterpolationMode.LANCZOS),  # 32, 32
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize(0.061, 0.140)
    ])
    # weak_transform = transforms.Compose([
    #     transforms.Resize(size=(256, 256), interpolation=InterpolationMode.LANCZOS),  # 32, 32
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor(),
    #     #transforms.Normalize(0.061, 0.140)
    # ])
    # strong_transform = transforms.Compose([
    #     transforms.Resize(size=(256, 256), interpolation=InterpolationMode.LANCZOS),  # 32, 32
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.RandomRotation(15),
    #     RandAugmentMC(n=2, m=10),
    #     CustomAugment(n=2, window_size=48),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(0.061, 0.140)
    # ])
    CAMUS_data_path = '/home/CV/AS/CAMUS'
    labeled_dataset = datasets.ImageFolder(labeled_data_path, transform=transform_labeledtrain)
    #labeled_dataset = datasets.ImageFolder(CAMUS_data_path, transform=transform_labeledtrain)
    # #unlabeled_dataset = datasets.ImageFolder(unlabeled_data_path,transform=TransformTwice(weak_transform, strong_transform))
    # unlabeled_dataset1= datasets.ImageFolder(unlabeled_data_path, transform=weak_transform)    #tensor([0.0630, 0.0630, 0.0630]) tensor([0.1332, 0.1332, 0.1332])
    # unlabeled_dataset2= datasets.ImageFolder(unlabeled_data_path, transform=strong_transform)
    # val_dataset = datasets.ImageFolder(val_data_path, transform=transforms.Compose([transforms.Resize(size=(256, 256), interpolation=InterpolationMode.LANCZOS),transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]))
    #
    mean, std=get_mean_and_std(labeled_dataset)
    print(mean, std)'''


    # 定义设备
    '''device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 定义模型
    classes = 5
    model = fastvit_t8(pretrained=False)
    num = model.head.in_features
    model.head = nn.Linear(num, classes)
    model.to(device)

    # 加载预训练模型参数
    resume = '/home/CV/AS/view/softmatch_5classes2/best_epoch_129_acc_86.3687_model.pth'
    state_dict = torch.load(resume, map_location=device)
    model.load_state_dict(state_dict)

    # 定义数据转换
    transform_test = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # 定义unlabeled数据集
    unlabeled_path = '/home/CV/AS/TMED/unlabeled/'
    dataset_test = datasets.ImageFolder(unlabeled_path, transform=transform_test)
    test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)
    num_batches=len(test_loader)
    # 创建保存预测结果的目录
    output_dir = '/home/CV/AS/TMED/PLAXPASX_byfastvit'
    os.makedirs(output_dir, exist_ok=True)

    # 用于存储预测标签为3或4的图片文件名的列表
    predicted_files = []

    model.eval()

    # 打开用于保存结果的txt文件
    txt_file_path = os.path.join(output_dir, 'predicted_images.txt')
    with open(txt_file_path, 'w') as txt_file:
        # 遍历测试数据集
        for batch_idx, (inputs, _) in enumerate(tqdm(test_loader, desc='Inference', unit='batch')):
            inputs = inputs.to(device)

            # 进行模型推理
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 获取预测标签值为3或4的图片名称
            for i in range(len(predicted)):
                if predicted[i] == 3 or predicted[i] == 4:
                    image_path, label = dataset_test.samples[batch_idx * test_loader.batch_size + i]
                    image_name = os.path.basename(image_path)
                    txt_file.write(image_name + '\n')

                    # 复制图片到保存预测结果的目录
                    src_path = os.path.join(unlabeled_path, 'unlabeled_set',image_name)
                    dest_path = os.path.join(output_dir, image_name)
                    shutil.copy(src_path, dest_path)

    # 关闭模型
    model.train()

    print("预测结果已保存到", txt_file_path)
    print("相应图片已复制到", output_dir)'''

