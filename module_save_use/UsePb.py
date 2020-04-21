import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
pb_path = 'log/FirstPB.pb'

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    input_data = np.random.rand(2,3)
    with open(pb_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            input_tensor = sess.graph.get_tensor_by_name("input:0")
            output_tensor = sess.graph.get_tensor_by_name("MatMul:0")
            output = sess.run(output_tensor,feed_dict={input_tensor:input_data})
            print("output = {0}".format(output))





