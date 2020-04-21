import tensorflow as tf
import numpy as np
import cv2
import os
import os.path as osp




from tensorflow.python.framework import graph_util
pb_path = 'E:\\deep_learn\\model\\catOrdog\\CatOrDog.pb'
valide_path = 'G:\\DATESETS\\VOCdevkit\\test1'


def get_im_info(_data_dir):
    """
    :return: 图片路径，图片标签
    """
    tem_indexs = []
    tem_labels = []
    _ims_filename = os.listdir(_data_dir)
    #cv2.namedWindow('ReadImage')
    for im_filename in _ims_filename:
        print(im_filename)
        if im_filename.find('cat') > -1:
            tem_labels.append([0, 1])
            print('Insert cat')
        elif im_filename.find('dog') > -1:
            tem_labels.append([1, 0])
            print('Insert dog')
        else:
            print('This is a What !')
        path = osp.join(_data_dir + '\\' + im_filename)
        # read_im = cv2.imread(path)
        # cv2.imshow('ReadImage',read_im)
        # cv2.waitKey(1000)

        tem_indexs.append(path)
    #, tem_labels
    return tem_indexs, tem_labels





with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    indexs, labels = get_im_info(valide_path)

    with open(pb_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_tensor = sess.graph.get_tensor_by_name("inputs:0")
            output_tensor = sess.graph.get_tensor_by_name("add_2:0")
            keep_drop = sess.graph.get_tensor_by_name("Placeholder_1:0")
            blobs = np.zeros((1, 244, 244, 3), dtype=np.float32)
            # cv2.namedWindow('CatOrDog')
            label_num = 0
            right_num = 0
            for index in indexs:
                im = cv2.imread(index)
                # cv2.imshow('CatOrDog', im)
                # cv2.waitKey(0)
                im = cv2.resize(im, (244, 244))
                im = im.astype(np.float32, copy=False)
                im -= 147
                blobs[0, :, :, :] = im

                output = sess.run(output_tensor, feed_dict={input_tensor: blobs, keep_drop: 1.0})
                probabilty = sess.run(tf.nn.softmax(output))
                # print("output = {0}".format(output))


                ret = probabilty.tolist()[0]
                if ret[0] <= ret[1] :#and labels[label_num][0] < labels[label_num][1]
                    right_num += 1
                    #print('It is a cat')
                elif ret[0] > ret[1] :#and labels[label_num][0] > labels[label_num][1]
                    right_num += 1
                    #print('It is a dog')

                label_num += 1
                ret = 'It is a cat' if ret[0] <= ret[1] else 'It is a dog'
                print('第 {0} 张图片 是 {1}'.format(label_num, ret))


                print('准确率 ： {:.3f}'.format(right_num / label_num))

                #
#Placeholder_1


