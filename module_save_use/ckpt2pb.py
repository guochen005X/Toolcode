import tensorflow as tf
from tensorflow.python.framework import graph_util

input_checkpoint = 'E:\\deep_learn\\model\\DogVsCat.ckpt'
output_node_name = 'add_2,Placeholder_1'
output_graph = ''
saver = tf.train.import_meta_graph('E:\\deep_learn\\model\\DogVsCat.ckpt.meta',clear_devices=True)
graph = tf.get_default_graph()
inout_graph_def = graph.as_graph_def()

with tf.Session() as sess:
    saver.restore(sess,input_checkpoint)
    output_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                                 input_graph_def=inout_graph_def,
                                                                 output_node_names=output_node_name.split(','))
    with tf.gfile.GFile('E:\\deep_learn\\model\\catOrdog\\CatOrDog.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    print("%d ops in the final graph." % len(output_graph_def.node))