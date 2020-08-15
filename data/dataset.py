# -*- coding:utf-8 -*-
# @time :2020.02.09
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from data import get_train_transform, get_test_transform

import sys 
sys.path.append("..") 
import cfg

from data import get_random_eraser


input_size = cfg.INPUT_SIZE
batch_size = cfg.BATCH_SIZE

# 构建数据提取器，利用dataloader
# 利用torchvision中的transforms进行图像预处理
#cfg为config文件，保存几个方便修改的参数

class  SelfCustomDataset(Dataset):
    def __init__(self, label_file, imageset):
        '''
        img_dir: 图片路径：img_dir + img_name.jpg构成图片的完整路径      
        '''
        # 所有图片的绝对路径
        with open(label_file, 'r') as f:
            #label_file的格式， （label_file image_label)
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
      # 相关预处理的初始化
      #   self.transforms=transform
        self.img_aug=True
        if imageset == 'train':
            self.transform= get_train_transform(size=cfg.INPUT_SIZE)
        else:
            self.transform = get_test_transform(size = cfg.INPUT_SIZE)
        self.eraser = get_random_eraser( s_h=0.1, pixel_level=True)
        self.input_size = cfg.INPUT_SIZE

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.img_aug:
            img =self.transform(img)


        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label)))
 
    def __len__(self):
        return len(self.imgs)



##ImageFolder对象可以将一个文件夹下的文件构造成一类
#所以数据集的存储格式为一个类的图片放置到一个文件夹下
#然后利用dataloader构建提取器，每次返回一个batch的数据，在很多情况下，利用num_worker参数
#设置多线程，来相对提升数据提取的速度

train_label_dir = cfg.TRAIN_LABEL_DIR
train_datasets = SelfCustomDataset(train_label_dir, imageset='train')
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

val_label_dir = cfg.VAL_LABEL_DIR
val_datasets = SelfCustomDataset(val_label_dir, imageset='test')
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=2)


##进行数据提取函数的测试
if __name__ =="__main__":

    for images, labels in train_dataloader:
        print(labels)
