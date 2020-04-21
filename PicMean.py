import cv2
import numpy as np
import os
import os.path as osp

def computeAllPicMean(picPaths,size,reshape = True):
    '''
    :param picPaths: 图片路径，以列表的形式
    :param size: 图片如果需要缩放，size为缩放的尺寸
    :return:三通道的均值，列表的形式
    '''
    channel0 = 0.0
    channel1 = 0.0
    channel2 = 0.0
    # cv2.namedWindow('originImg')
    # cv2.namedWindow('newImg')
    for picPath in picPaths:
        pic = cv2.imread(picPath)
        #if reshape:
        pic = cv2.resize(pic, (size[0],size[1]))
        # cv2.imshow('originImg',pic)
        pic -= 147
        # cv2.imshow('newImg',pic)
        # cv2.waitKey(30)
        channel0 += np.mean(pic[:,:,0])
        channel1 += np.mean(pic[:,:,1])
        channel2 += np.mean(pic[:,:,2])

    mean0 = channel0 / len(picPaths)
    mean1 = channel1 / len(picPaths)
    mean2 = channel2 / len(picPaths)
    mean = []
    mean.append(mean0)
    mean.append(mean1)
    mean.append(mean2)
    print('B_mean = {}  G_mean = {} R_mean = {}'.format(mean[0], mean[1], mean[2]))
    #mean = np.array([mean0,mean1, mean2], np.uint8)
    return mean

if __name__ == '__main__':
    #imgPaths = os.listdir("G:\\DATESETS\\64_CASIA-FaceV5\\data\\000")
    imgPaths = os.listdir("G:\\DATESETS\\VOCdevkit\\train")
    completePaths = list()
    for imgPath in imgPaths:
        imgPath = osp.join("G:\\DATESETS\\VOCdevkit\\train",imgPath)
        completePaths.append(imgPath)
    imgSize = [244,244]
    means = computeAllPicMean(completePaths, imgSize)
    print(means)