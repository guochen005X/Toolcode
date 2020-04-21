import numpy as np
import cv2
from GenerateLabel import From_dir_get_label
from PicMean import computeAllPicMean

class TrainData:
    def __init__(self, minibatch = 8):
        print('Init Train Data')
        self._minibatch = minibatch
        self._start = 0
        #self._end = self._start + minibatch

    def generateLabelsAPath(self, dir):
        self._dataPaths, self._dataLabels, self._numClass = From_dir_get_label(dir)
        print('generate Path And Label is finish !')
        return self._dataPaths, self._dataLabels, self._numClass

    def SetMean(self,mean):
        self._mean = mean

    def getNextMinibatch(self, start):
        blobs = np.zeros((self._minibatch, 224, 224, 3), np.float32)
        labels = np.zeros(self._minibatch, np.float32)
        end = self._start + self._minibatch
        if end >= len(self._dataLabels):
            end = len(self._dataLabels)
        # cv2.namedWindow('facePicture')
        for index in range(self._start,end):
            dataPath = self._dataPaths[index]
            img = cv2.imread(dataPath)
            img = cv2.resize(img, (224,224))
            img -= self._mean
            # cv2.imshow('facePicture',img)
            # cv2.waitKey(10)

            img = img.astype(np.float32, copy=False)
            #减去均值

            blobs[index,:,:,:] = img
            labels[index] = self._dataLabels[index]
        #cv2.destroyWindow('facePicture')
        self._start += self._minibatch
        return blobs, labels

    def getRandomMinibatch(self):
        blobs = np.zeros((self._minibatch, 224, 224, 3), np.float32)
        labels = np.zeros(self._minibatch, np.float32)
        randomIndex = np.random.randint(0,labels.shape[0], self._minibatch)
        for i in range(randomIndex.shape[0]):
            dataPath = self._dataPaths[randomIndex[i]]
            img = cv2.imread(dataPath)
            img = cv2.resize(img, (224, 224))
            img -= self._mean
            img = img.astype(np.float32, copy=False)
            blobs[i, :, :, :] = img
            labels[i] = self._dataLabels[randomIndex[i]]
        return blobs, labels


#mean 88 67 61
if __name__ == '__main__':
    trainData = TrainData()
    dataPaths, dataLabels, numClass = trainData.generateLabelsAPath("G:\\DATESETS\\64_CASIA-FaceV5\\data")
    trainData.SetMean(np.array([[88, 67, 61]], np.uint8))
    for i in range(int(2500/16)):
        blobs, labels = trainData.getNextMinibatch(i)
        i += 16
    print('Get All img and label !')
