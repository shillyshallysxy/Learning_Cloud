import tensorflow as tf
a = tf.constant([[[1, 2, 3], [3, 2, 1]]])
# b = tf.constant([[1, 2]])
c = tf.constant([[1, 2, 3]])
d,e,f = a.get_shape().as_list()
print(d,e)
with tf.Session() as sess:
    print(sess.run(tf.tile(a, [3, 1, 1])))
    # print(sess.run(tf.multiply(a, b)))
