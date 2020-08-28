# -*- coding:utf-8 -*-
# @time :2020/8/21
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com


from PIL import Image

import io
from flask import Flask, request, jsonify
import time
import json
import uuid

import sys
sys.path.append('.')

from cfg import *
from utils import db, image_transform, base64_encode_image

# 初始化实例
app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():

    data = {'Success': False}

    if request.files.get('image'):

        now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

        image = request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        image = image_transform(InputSize)(image).numpy()
        # 将数组以C语言存储顺序存储
        image = image.copy(order="C")
        # 生成图像ID
        k = str(uuid.uuid4())
        d = {"id": k, "image": base64_encode_image(image)}
        # print(d)
        db.rpush(ImageQueue, json.dumps(d))
        # 运行服务
        while True:
            # 获取输出结果
            output = db.get(k)
            # print(output)
            if output is not None:
                output = output.decode("utf-8")
                data["predictions"] = json.loads(output)
                db.delete(k)
                break
            time.sleep(ClientSleep)
        data["success"] = True
    return jsonify(data)

if __name__ == '__main__':

    app.run(host='127.0.0.1', port =5000,debug=True )





