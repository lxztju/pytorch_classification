
## 简介

基于torchision实现的pytorch图像分类功能。


## 近期更新

* 2022.11.05更新
    - 新添加tensorrt c++的推理方案

* 2022.10.29更新，进行代码重构，基本的功能基本一致。
    - 支持pytorch ddp的训练
    - 支持c++ libtorch的模型推理
    - 支持script脚本一键运行
    - 添加日志模块

习惯之前版本的请看v1版本的代码：[V1版本](https://github.com/lxztju/pytorch_classification/tree/v1)。


主要功能：

利用pytorch实现图像分类，基于torchision可以扩展使用densenet，resnext，mobilenet，efficientnet，swin transformer等图像分类网络

如果有用欢迎star

## 实现功能
* 基础功能利用pytorch实现图像分类
* 包含带有warmup的cosine学习率调整
* warmup的step学习率优调整
* 多模型融合预测，加权与投票融合
* 利用flask + redis实现模型云端api部署（tag v1）
* c++ libtorch的模型部署
* 使用tta测试时增强进行预测（tag v1）
* 添加label smooth的pytorch实现（标签平滑）（tag v1）
* 添加使用cnn提取特征，并使用SVM，RF，MLP，KNN等分类器进行分类（tag v1）。
* 可视化特征层

## 运行环境
* python3.7
* pytorch 1.8.1
* torchvision 0.9.1
* opencv(libtorch cpp推理使用， 版本3.4.6)（可选）
* libtorch cpp推理使用（可选）



## 快速开始

### 数据集形式
 数据集的组织形式，参考[sample_files/imgs/listfile.txt](https://github.com/lxztju/pytorch_classification/blob/master/sample_files/imgs/listfile.txt)


### 训练 测试

修改`run.sh`中的参数，直接运行run.sh即可运行


主要修改的参数：

```
OUTPUT_PATH 模型保存和log文件的路径

TRAIN_LIST 训练数据集的list文件
VAL_LIST  测试集合的list文件
model_name 默认是resnet50
lr 学习率
epochs 训练总的epoch
batch-size  batch的大小
j dataloader的num_workers的大小
num_classes 类别数
```


### libtorch inference


代码存储在`cpp_inference`文件夹中。

1. 利用[cpp_inference/traced_model/trace_model.py](https://github.com/lxztju/pytorch_classification/blob/master/cpp_inference/traced_model/trace_model.py)将训练好的模型导出。
2. 编译所需的opencv和libtorch代码到`cpp_inference/third_party_library`

3. 编译
```
sh compile.sh
```

4. 可执行文件测试
```
./bin/imgCls imgpath
```


