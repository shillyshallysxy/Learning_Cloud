import tensorflow as tf
from Transformer.transformer_utils import *
a = tf.constant([[[1, 2, 3], [3, 2, 1]]])
# b = tf.constant([[1, 2]])
c = tf.constant([[1, 2, 3]])
inpt = tf.placeholder(shape=[None, None], dtype=tf.float32)
p1 = inpt.get_shape()
d,e,f = a.get_shape().as_list()
print(d,e)
with tf.Session() as sess:
    print(sess.run(tf.pad(tf.expand_dims(tf.to_float(tf.range(5)), 1), [[0,0], [1,1]])))
    vvv = get_timing_signal_1d(10, 10)
    vv = sess.run(vvv)
    print(vv)
    # print(sess.run(tf.multiply(a, b)))

a = [1, 2, 3]
a = a+[2]*3
print(1)