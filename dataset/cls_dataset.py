import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import numpy as np

class ImageInfo:
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])



class  ClsDataset(Dataset):
    def __init__(self, list_file, transform=None):
        self.list_file = list_file
        self.transform = transform
        
        self._parser_input_data()
                 
    def _parser_input_data(self):
        assert os.path.exists(self.list_file)
        
        lines = [x.strip().split('\t') for x in open(self.list_file, encoding='utf-8')]
        
        self.imgs_list = [ImageInfo(line) for line in lines]
        
        

    def __getitem__(self, index):
        img_info = self.imgs_list[index]
    
        img = Image.open(img_info.path).convert('RGB')
        if self.transform != None:
            img =self.transform(img)
        return img, torch.as_tensor(img_info.label, dtype=torch.long), img_info.path
 
    def __len__(self):
        return len(self.imgs_list)

if __name__ == '__main__':
    clsdataset = ClsDataset(list_file='list.txt')
    for k, v in clsdataset:
        print(k, v)
