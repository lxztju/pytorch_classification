import os
home = os.path.expanduser('~')
## 预训练模型的存放位置
LOCAL_PRETRAINED = {
    'resnext101_32x8d': home + '/weights/resnext101_32x8.pth',
    'resnext101_32x16d': home + '/weights/resnext101_32x16.pth',
    'resnext101_32x48d': home + '/weights/resnext101_32x48.pth',
    'resnext101_32x32d': home + '/weights/resnext101_32x32.pth',
    'resnet50': home +'/weights/resnet50.pth',
    'resnet101': home +'/weights/resnet101.pth',
    'densenet121': home +'/weights/densenet121.pth',
    'densenet169': home +'/weights/densenet169.pth',
    'moblienetv2': home +'/weights/mobilenetv2.pth',
    'efficientnet-b7': home + '/weights/efficientnet-b7.pth',
    'efficientnet-b8': home + '/weights/efficientnet-b8.pth'
}

model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'moblienetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}
from .vision import *
from .resnext_wsl import *
from .efficientnet_pytorch import *
from .build_model import *
