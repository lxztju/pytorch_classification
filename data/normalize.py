import sys
import os
import glob
import shutil

sys.path.append("..")
import cfg

Base = cfg.BASE + 'val'

labels = os.listdir(Base)  # 获取val目录下标签
labels.sort()  # 对标签排序

for index, label in enumerate(labels):
    print(index, label)
    img_list = glob.glob(os.path.join(Base, label, "*.*"))  # 在迭代中分别获取文件名列表
    for img in img_list:
        dst = Base + '/' + str(index) + '.' + os.path.basename(img)  # 按照"predict.py"中的读取规则和当前图片所在标签重命名
        shutil.move(img, dst)
        print('<' + img + '> moved to <' + dst + '>')
    os.rmdir(Base + '/' + label)  # 删除经过处理的空目录

print("normalization done")
