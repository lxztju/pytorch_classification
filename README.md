# pytorch_classification

利用pytorch实现图像分类，其中包含的densenet，resnext，mobilenet，efficientnet, resnet等图像分类网络，可以根据需要再行利用torchvision扩展其他的分类算法

## 实现功能
* 基础功能利用pytorch实现图像分类
* 包含带有warmup的cosine学习率调整
* warmup的step学习率优调整
* 多模型融合预测，加权与投票融合
* 利用flask实现模型云端api部署
* 使用tta测试时增强进行预测
* 添加label smooth的pytorch实现（标签平滑）
* 添加使用cnn提取特征，并使用SVM，RF，MLP，KNN等分类器进行分类。
* 更新添加了模型蒸馏的的训练方法
* 添加中间层可视化
* 更新模型部署(采用flask+Redis的方法)
* c++ libtorch进行模型部署的简单demo

## 运行环境
* python3.7
* pytorch 1.1
* torchvision 0.3.0

## 代码仓库的使用

### 数据集形式
原始数据集存储形式为，同个类别的图像存储在同一个文件夹下，所有类别的图像存储在一个主文件夹data下。

```
|-- data
    |-- train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label    
            |--*.jpg
        ...

    |-- val
        |--*.jpg
```

利用preprocess.py将数据集格式进行转换（个人习惯这种数据集的方式）

```
python ./data/preprocess.py
```

转换后的数据集为，将训练集的路径与类别存储在train.txt文件中，测试机存储在val.txt中.
其中txt文件中的内容为

```
# train.txt

/home/xxx/data/train/label1/*.jpg   label

# val.txt

/home/xxx/data/train/label1/*.jpg
```

```
|-- data
    |-- train
        |--label1
            |--*.jpg
        |--label2
            |--*.jpg
        |--label    
            |--*.jpg
        ...

    |-- val
        |--*.jpg
    |--train.txt
    |--val.txt
```


### 模型介绍
仓库中模型densenet，mobilenet,resnext模型来自于torchvision

efficientnet来自于 https://github.com/lukemelas/EfficientNet-PyTorch

### 训练

* 在`cfg.py`中修改合适的参数，并在train.py中选择合适的模型

```
##数据集的类别
NUM_CLASSES = 206

#训练时batch的大小
BATCH_SIZE = 32

#网络默认输入图像的大小
INPUT_SIZE = 300
#训练最多的epoch
MAX_EPOCH = 100
# 使用gpu的数目
GPUS = 2
# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
# 初始学习率
LR = 1e-3
# 训练好模型的保存位置
SAVE_FOLDER = './weights'

# 采用的模型名称
model_name = 'resnext101_32x32d'
```


```shell
python train.py
```

### 预测
在cfg.py中`TRAINED_MODEL`参数修改为指定的权重文件存储位置,在predict文件中可以选定是否使用tta

```shell
python predict.py
```

* 当训练完成多模型之后生成多个txt文件之后，利用`ensamble`文件夹中的`kaggle_vote.py`进行投票融合或者加权投票融合。

将每个模型生成的csv文件，移动到`ensamble/samples/`然后将每个文件命名为`method1.py`,`method2.py`.
然后运行如下命令进行投票融合

```shell
python ./kaggle_vote.py "./samples/method*.csv" "./samples/vote.csv"
```



### cnn + svm


代码存在于`cnn_ml.py`中, 利用训练好的cnn特征提取器，将得到的特征保存为pkl文件，然后训练svm分类器， 并将分类器模型保存，然后读取预测

主要需要修改的就是根据不同模型的输出特征向量的大小在`cnn_ml.py`中修改`NB_features`对应的大小

### flask云端部署

将训练存储好的权重文件，存储在`flask_deployment`文件夹中

然后修改`server.py`中路径运行即可
利用`client.py`进行调用


