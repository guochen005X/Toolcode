import cv2

import os
import os.path as osp

import numpy as np

Save_rootdir = 'E:\deep_learn\VGG16\Model_Persistence\Tool'

if __name__ == '__main__':
    cv2.namedWindow('capture')
    cap = cv2.VideoCapture(0)
    img_name = 'five'
    extra_name = '.jpg'
    index = 0
    mean = np.zeros(3, np.int64)
    while cap.isOpened():
        ok , im = cap.read()
        if not ok:
            break
        cv2.imshow('capture',im)
        print('Will Save a picture !')
        c = cv2.waitKey(10)
        save_name = img_name + str(index) + extra_name
        save_dir = osp.join(Save_rootdir,save_name)
        cv2.imwrite(save_dir, im)
        mean = np.sum(im, axis=(0, 1)).astype(int)/(im.shape[0]  * im.shape[1])
        print('mean = {0} h = {1}  w = {2}'.format(mean, im.shape[0], im.shape[1]))
        #print('Save ' + str(index) + ' Image !')
        index += 1
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow()