# -*- coding:utf-8 -*-
# @time :2020/8/21
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com


import numpy as np
from PIL import Image

from flask import Flask
from cfg import *
import redis
import base64
import sys

from torchvision import transforms


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def image_transform(inputsize):
    return transforms.Compose([
        Resize((int(inputsize[0] * (256 / 224)), int(inputsize[1] * (256 / 224)))),
        transforms.CenterCrop(inputsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def decode_predictions(predition):
    return predition




# 构建redis数据库
db = redis.StrictRedis(host="127.0.0.1", port=6379, db=0)


def base64_encode_image(img):
    return base64.b64encode(img).decode("utf-8")

def base64_decode_image(img, dtype, shape):
    # 查看python版本,如果是python3版本进行转换
    if sys.version_info.major == 3:
        img = bytes(img, encoding="utf-8")
    img = np.frombuffer(base64.decodebytes(img), dtype=dtype)
    img = img.reshape(shape)
    return img



def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    # return the processed image
    return image