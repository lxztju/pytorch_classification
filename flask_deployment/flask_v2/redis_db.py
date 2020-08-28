# -*- coding:utf-8 -*-
# @time :2020/8/21
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com

import torch
import numpy as np
import time

import json
from cfg import  InputSize, ImageQueue, BatchSize, ImageType, ClientSleep, ServeSleep, Channel
from utils import db, base64_decode_image,decode_predictions




def classify_process(filepath):
    # 导入模型
    print("* Loading model...")
    model = load_checkpoint(filepath)
    print("* Model loaded")
    while True:
        # 从数据库中创建预测图像队列
        queue = db.lrange(ImageQueue, 0, BatchSize - 1)
        imageIDs = []
        batch = None
        # 遍历队列
        for q in queue:
            # 获取队列中的图像并反序列化解码
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], ImageType,
                                        (1, InputSize[0], InputSize[1], Channel))
            # 检查batch列表是否为空
            if batch is None:
                batch = image
            # 合并batch
            else:
                batch = np.vstack([batch, image])
            # 更新图像ID
            imageIDs.append(q["id"])
            # print(imageIDs)
        if len(imageIDs) > 0:
            print("* Batch size: {}".format(batch.shape))
            preds = model(torch.from_numpy(batch.transpose([0, 3,1,2])))
            results = decode_predictions(preds)
            # 遍历图像ID和预测结果并打印
            for (imageID, resultSet) in zip(imageIDs, results):
                # initialize the list of output predictions
                output = []
                # loop over the results and add them to the list of
                # output predictions
                print(resultSet)
                for label in resultSet:
                    prob = label.item()
                    r = {"label": label.item(), "probability": float(prob)}
                    output.append(r)
                # 保存结果到数据库
                db.set(imageID, json.dumps(output))
            # 从队列中删除已预测过的图像
            db.ltrim(ImageQueue, len(imageIDs), -1)
        time.sleep(ServeSleep)




def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


if __name__ == '__main__':
    filepath = '../c/resnext101_32x8.pth'
    classify_process(filepath)