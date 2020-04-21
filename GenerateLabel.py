import numpy as np
import os
import os.path as osp

#返回的标签对应十进制，没有转换成0、1样式
def From_dir_get_label(dir_name):
    assert osp.exists(dir_name) , 'dir_name Is Empty !'
    filenames = os.listdir(dir_name)
    class_num = len(filenames)
    filename_labels = zip(filenames, range(len(filenames)))
    imagesLabels = list()
    for filename_label in filename_labels:
        imgNames = os.listdir( osp.join(dir_name,filename_label[0] ) )
        #imagesLabels = []
        for imgName in imgNames:
            imgPathLabel = []
            imgPath = osp.join(dir_name,filename_label[0], imgName)
            imgPathLabel.append(imgPath)
            imgPathLabel.append(filename_label[1])
            imagesLabels.append(imgPathLabel)

    pre_index = np.arange(len(imagesLabels))
    np.random.shuffle(pre_index)
    indexs = []
    labels = []
    for index in range(pre_index.shape[0]):
        indexs.append(imagesLabels[pre_index[index]][0])
        labels.append(imagesLabels[pre_index[index]][1])

    for indexLabel in zip(indexs, labels) :
        print(indexLabel)

    return indexs, labels, class_num


if __name__ == '__main__':
    indexs, labels, class_num = From_dir_get_label("G:\\DATESETS\\64_CASIA-FaceV5\\data")
    binary_labels = np.zeros(shape=[len(labels), class_num], dtype=np.float32)

    for index in range(len(labels)):
        binary_labels[index][labels[index]] = 1
    for label_one in binary_labels:
        print(label_one)
    print('行{0}   列{1}'.format(binary_labels.shape[0], binary_labels.shape[1]))



