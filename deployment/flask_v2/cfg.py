# -*- coding:utf-8 -*-
# @time :2020/8/21
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com


'''
定义web服务所需的一些配置参数
'''

# 输入图像的大小
InputSize = (224, 224)

# 输入图像的通道数
Channel = 3

# 输入处理图像的batch
BatchSize = 16

# 数据类型
ImageType = 'float32'

# 模型运行设备
Device = 'cpu'

# redis运行的图像队列
ImageQueue = 'image_queue'

ServeSleep = 0.1
ClientSleep = 0.1


