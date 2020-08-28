# -*- coding:utf-8 -*-
# @time :2020/8/21
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju
# @Emial : lxztju@163.com


# curl -X POST -F image=@test.jpg 'http://127.0.0.1:5000/predict'

from threading import Thread
import requests
import time

# 请求的URL
REST_API_URL = "http://127.0.0.1:5000/predict"
# 测试图片
IMAGE_PATH = "./test.jpg"

# 并发数
NUM_REQUESTS = 500
# 请求间隔
SLEEP_COUNT = 0.05
def call_predict_endpoint(n):

    # 上传图像
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}
    # 提交请求
    r = requests.post(REST_API_URL, files=payload).json()
    # 确认请求是否成功
    if r["success"]:
        print("[INFO] thread {} OK".format(n))
    else:
        print("[INFO] thread {} FAILED".format(n))
# 多线程进行
for i in range(0, NUM_REQUESTS):
    # 创建线程来调用api
    t = Thread(target=call_predict_endpoint, args=(i,))
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)
time.sleep(300)