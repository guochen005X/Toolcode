import tensorflow as tf
import numpy as np





if __name__ == '__main__':

    input_data = tf.placeholder(dtype=tf.float32, shape=[2,3],name='input')
    print("input_data.node_name = " + input_data.name)

    p1 = tf.Variable(initial_value=tf.random_normal(shape=[2,3], mean=1.0, stddev=0.5), name='v1')
    p2 = tf.Variable(initial_value=tf.random_normal(shape=[2,3], mean=1.0, stddev=0.5), name='v2')
    pinput = input_data + p1
    p3 = pinput + p2
    print("p3.node_name = " + p3.name)
    W = tf.Variable(initial_value=tf.random_normal(shape=[3,2], mean=0.3, stddev=0.2), name='w')
    wp = tf.matmul(p3, W)
    print("wp.node_name = " + wp.name)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.save(sess, 'log/model.ckpt')

        ckpt = tf.train.get_checkpoint_state('log')
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)