import os
import tensorflow as tf
import cv2
import numpy as np
import csv

data_folder_name = '..\\temp'
data_path_name = 'cv'
pic_path_name = 'pic'
cv_path_name = 'fer2013'
csv_file_name = 'fer2013.csv'
model_path_name = 'cnn_inuse'
ckpt_name = ['cnn_emotion_classifier.ckpt',
             'cnn_emotion_classifier_boost.ckpt',
             'cnn_emotion_classifier_boost2.ckpt',
             'cnn_emotion_classifier_boost3.ckpt',
             'cnn_emotion_classifier_boost4.ckpt',
             'cnn_emotion_classifier_g.ckpt', ]
# ckpt_name = ['cnn_emotion_classifier.ckpt']
save_ckpt_name = 'cnn_emotion_classifier_stacking.ckpt'
# record_name_train = 'fer2013_train.tfrecord'
record_name_train = 'fer2013_test.tfrecord'
record_name_test = 'fer2013_test.tfrecord'
record_name_eval = 'fer2013_eval.tfrecord'
classifier_num = len(ckpt_name)
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
csv_path = os.path.join(cv_path, csv_file_name)
ckpt_path = []
for ckpt_n_ in ckpt_name:
    ckpt_path.append(os.path.join(data_folder_name, data_path_name, model_path_name, ckpt_n_))
save_ckpt_path = os.path.join(data_folder_name, data_path_name, model_path_name, save_ckpt_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)

channel = 1
default_height = 48
default_width = 48
batch_size = 256
test_batch_size = 3589
shuffle_pool_size = 4000
generations = 100
save_flag = True
retrain = False
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)


# 数据增强
def pre_process_img(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32./255)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.random_crop(image, [default_height-np.random.randint(0, 4), default_width-np.random.randint(0, 4), 1])
    return image


def __parse_function_csv(serial_exmp_):
    features_ = tf.parse_single_example(serial_exmp_,
                                        features={"image/label": tf.FixedLenFeature([], tf.int64),
                                                  "image/height": tf.FixedLenFeature([], tf.int64),
                                                  "image/width": tf.FixedLenFeature([], tf.int64),
                                                  "image/raw": tf.FixedLenFeature([default_width*default_height*channel]
                                                                                  , tf.int64)})
    label_ = tf.cast(features_["image/label"], tf.int32)
    height_ = tf.cast(features_["image/height"], tf.int32)
    width_ = tf.cast(features_["image/width"], tf.int32)
    image_ = tf.cast(features_["image/raw"], tf.int32)
    image_ = tf.reshape(image_, [height_, width_, channel])
    image_ = tf.multiply(tf.cast(image_, tf.float32), 1. / 255)
    # image_ = pre_process_img(image_)
    image_ = tf.image.resize_images(image_, [default_height, default_width])
    return image_, label_


def get_dataset(record_name_):
    record_path_ = os.path.join(data_folder_name, data_path_name, record_name_)
    data_set_ = tf.data.TFRecordDataset(record_path_)
    return data_set_.map(__parse_function_csv)


def pred_proba(images_, sess_, x_in_, drop_, logit_):
    pred_prop_ = sess_.run([tf.nn.softmax(logit_)], {x_in_: images_, drop_: 1.0})
    return pred_prop_


def pred_proba_mul(images_, sess_, x_in_, drop_, logit_):
    pred_prob_mul_ = []
    for i_, session_ in enumerate(sess_):
        pred_prob_mul_.append(pred_proba(images_, session_, x_in_[i_], drop_[i_], logit_[i_]))
    return np.squeeze(np.array(pred_prob_mul_), axis=1)


def evaluate(logits_, y_):
    return np.mean(np.equal(np.argmax(logits_, axis=1), y_))


def main(argv):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        data_set_train = get_dataset(record_name_train)
        data_set_train = data_set_train.shuffle(shuffle_pool_size).batch(batch_size).repeat()
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle())

        data_set_test = get_dataset(record_name_test)
        data_set_test = data_set_test.batch(test_batch_size).repeat()
        data_set_test_iter = data_set_test.make_one_shot_iterator()
        test_handle = sess.run(data_set_test_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[], name='handle')
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types, data_set_train.output_shapes)
        x_input_bacth, y_target_batch = iterator.get_next()

        classifier_weight = tf.Variable([[1]*classifier_num], dtype=tf.float32, name='classifier_weight')
        t_x_input = tf.placeholder(dtype=tf.float32, shape=[classifier_num, None, num_class])
        t_y_target = tf.placeholder(dtype=tf.int32, shape=[None])
        t_x_reshape = tf.reshape(t_x_input, [classifier_num, -1])
        model_output = tf.reshape(tf.matmul(classifier_weight, t_x_reshape), [-1, num_class])

        # model_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output,
        #                                                                            labels=t_y_target))
        model_output = tf.div(model_output, tf.reduce_sum(classifier_weight))
        t_y_target_dense = tf.expand_dims(t_y_target, 1)
        index = tf.expand_dims(tf.range(0, tf.shape(t_y_target)[0]), 1)
        concated = tf.concat([index, t_y_target_dense], 1)
        one_hot = tf.sparse_to_dense(concated, [tf.shape(t_y_target)[0], num_class], 1., 0.)
        model_loss = tf.reduce_sum(tf.square(tf.subtract(one_hot, model_output)))

        model_train_step = tf.train.AdamOptimizer(0.01).minimize(model_loss, var_list=[classifier_weight])
        sess.run(tf.global_variables_initializer())
        if retrain:
            saver = tf.train.Saver({"classifier_weight": classifier_weight})
            saver.restore(sess, save_ckpt_path)
        else:
            pass

        sessions = []
        graphs = []
        for i in range(classifier_num):
            graph_temp = tf.Graph()
            graphs.append(graph_temp)
            sessions.append(tf.Session(graph=graph_temp))
        x_input = []
        dropout = []
        logits = []
        for graph_, sess_, c_p_ in zip(graphs, sessions, ckpt_path):
            with sess_.as_default():
                with graph_.as_default():
                    saver = tf.train.import_meta_graph(c_p_ + '.meta')
                    saver.restore(sess_, c_p_)
                    x_input.append(graph_.get_tensor_by_name('x_input:0'))
                    dropout.append(graph_.get_tensor_by_name('dropout:0'))
                    logits.append(graph_.get_tensor_by_name('project/output/logits:0'))

        print('start training')
        saver = tf.train.Saver({"classifier_weight": classifier_weight}, max_to_keep=1)
        max_accuracy = 0
        temp_train_loss = []
        temp_test_loss = []
        temp_train_acc = []
        temp_test_acc = []
        for i in range(generations):
            x_batch, y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: train_handle})
            pred_prob_mul = pred_proba_mul(x_batch, sessions, x_input, dropout, logits)
            # print(pred_prob_mul)
            train_feed_dict = {t_x_input: pred_prob_mul, t_y_target: y_batch}
            sess.run(model_train_step, train_feed_dict)
            if (i + 1) % 1 == 0:
                train_loss, train_logits = sess.run([model_loss, model_output], train_feed_dict)
                train_accuracy = evaluate(train_logits, y_batch)
                print('Generation # {}. Train Loss : {:.3f} . '
                      'Train Acc : {:.4f}'.format(i, train_loss, train_accuracy))
                temp_train_loss.append(train_loss)
                temp_train_acc.append(train_accuracy)
                print(sess.run(classifier_weight))
            if (i + 1) % 1 == 0:
                test_x_batch, test_y_batch = sess.run([x_input_bacth, y_target_batch], feed_dict={handle: test_handle})
                pred_prob_mul_test = pred_proba_mul(test_x_batch, sessions, x_input, dropout, logits)
                test_feed_dict = {t_x_input: pred_prob_mul_test, t_y_target: test_y_batch}
                test_loss, test_logits = sess.run([model_loss, model_output], test_feed_dict)
                test_accuracy = evaluate(test_logits, test_y_batch)
                print('Generation # {}. Test Loss : {:.3f} . '
                      'Test Acc : {:.4f}'.format(i, test_loss, test_accuracy))
                temp_test_loss.append(test_loss)
                temp_test_acc.append(test_accuracy)
                if test_accuracy >= max_accuracy and save_flag:
                    max_accuracy = test_accuracy
                    saver.save(sess, os.path.join(data_folder_name, data_path_name, save_ckpt_name))
                    print('Generation # {}. --model saved--'.format(i))
        print('Last accuracy : ', max_accuracy)


if __name__ == '__main__':
    tf.app.run()