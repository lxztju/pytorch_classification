# -*- coding:utf-8 -*-
# @time :2020.06.28
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju

import os
import torch
import torch.nn as nn
import torch.optim as optim

from data import train_dataloader,train_datasets
import cfg
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step, loss_fn_kd


##创建训练模型参数保存的文件夹
save_folder = cfg.SAVE_FOLDER + cfg.model_name
os.makedirs(save_folder, exist_ok=True)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model


def train_kd(model, teacher_model, optimizer, loss_fn_kd, T, alpah):

    # set student model to training mode
    model.train()
    teacher_model.eval() 

    lr = cfg.LR

    batch_size = cfg.BATCH_SIZE
    #每一个epoch含有多少个batch
    max_batch = len(train_datasets)//batch_size
    epoch_size = len(train_datasets) // batch_size
    ## 训练max_epoch个epoch
    max_iter = cfg.MAX_EPOCH * epoch_size

    start_iter = cfg.RESUME_EPOCH * epoch_size

    epoch = cfg.RESUME_EPOCH

    # cosine学习率调整
    warmup_epoch=5
    warmup_steps = warmup_epoch * epoch_size
    global_step = 0

    # step 学习率调整参数
    stepvalues = (10 * epoch_size, 20 * epoch_size, 30 * epoch_size)
    step_index = 0

    for iteration in range(start_iter, max_iter):
        global_step += 1

        ##更新迭代器
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(train_dataloader)
            loss = 0
            epoch += 1
            ###保存模型
            if epoch % 5 == 0 and epoch > 0:
                if cfg.GPUS > 1:
                    checkpoint = {'model': model.module,
                                'model_state_dict': model.module.state_dict(),
                                # 'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
                else:
                    checkpoint = {'model': model,
                                'model_state_dict': model.state_dict(),
                                # 'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch}
                    torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))

        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate_step(optimizer, cfg.LR, 0.1, epoch, step_index, iteration, epoch_size)

        ## 调整学习率
        # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
        #                           learning_rate_base=cfg.LR,
        #                           total_steps=max_iter,
        #                           warmup_steps=warmup_steps)


        ## 获取image 和 label
        # try:
        images, labels = next(batch_iterator)
        # except:
        #     continue

        ##在pytorch0.4之后将Variable 与tensor进行合并，所以这里不需要进行Variable封装
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
            teacher_model = teacher_model.cuda()
            model = model.cuda()
        teacher_outputs = teacher_model(images)
        out = model(images)
        loss = loss_fn_kd(out, labels, teacher_outputs,T, alpha)

        optimizer.zero_grad()  # 清空梯度信息，否则在每次进行反向传播时都会累加
        loss.backward()  # loss反向传播
        optimizer.step()  ##梯度更新

        prediction = torch.max(out, 1)[1]
        train_correct = (prediction == labels).sum()
        ##这里得到的train_correct是一个longtensor型，需要转换为float
        # print(train_correct.type())
        train_acc = (train_correct.float()) / batch_size

        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))

 

if __name__ == '__main__':
    save_folder ='./weights/epoch_30.pth'
    teacher_model =  load_checkpoint(save_folder)


    student_model_name = 'moblienetv2'
    student_model = cfg.MODEL_NAMES[student_model_name](num_classes=cfg.NUM_CLASSES)

    ##定义优化器与损失函数
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    optimizer = optim.SGD(student_model.parameters(), lr=cfg.LR,
                        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    T = 10
    alpha = 0.5
    train_kd(student_model, teacher_model, optimizer, loss_fn_kd, T, alpha)