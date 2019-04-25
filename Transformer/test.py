import tensorflow as tf
sess = tf.Session()
# from Transformer.transformer_utils import *
# a = tf.Variable([[1, 2, 3], [3, 2, 1]], dtype=tf.float32)
# # b = tf.constant([[1, 2]])
# c = tf.reshape(tf.to_float([[1, 2, 3]]), [3, 1])
# loss = tf.reduce_sum(tf.subtract(tf.matmul(a, c), [[1], [1]]))
# opt = tf.train.GradientDescentOptimizer(0.001)
# grads = opt.compute_gradients(loss)
# var_list = tf.trainable_variables()
# if var_list[0] == grads[0][1]:
#     print(11112321321)
#
#     # print(sess.run(tf.multiply(a, b)))
#
# layer_name_list = ["encoder_layer_"+str(i) for i in range(4)]
# print(layer_name_list)
# a = [1, 2,3 ]
# for ii, i in enumerate(a):
#     if i == 1:
#         a[ii] = 2
# print(a)
# if "abc" in "abcdefg":
#     print("sdadsadsa")
a = tf.to_int32(10)
b = tf.range(a)
c = sess.run(b)
for i in range(2):
    print(i)
for i in range(1, 2):
    print(i)