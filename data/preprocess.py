import os
import glob



import sys 
sys.path.append("..") 
import cfg




if __name__ == '__main__':
    traindata_path = cfg.BASE + 'train'
    labels = os.listdir(traindata_path)
    valdata_path = cfg.BASE + 'val'
    ##写train.txt文件
    txtpath = cfg.BASE
    # print(labels)
    for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(traindata_path,label, '*.*'))
        # print(imglist)
        with open(txtpath + 'train.txt', 'a')as f:
            for img in imglist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')
        # print(imglist)

    imglist = glob.glob(os.path.join(valdata_path, '*.*.*'))
    with open(txtpath + 'val.txt', 'a')as f:
        for img in imglist:
            f.write(img)
            f.write('\n')