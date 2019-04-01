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
record_name_train = 'fer2013_train.tfrecord'
record_name_test = 'fer2013_test.tfrecord'
record_name_eval = 'fer2013_eval.tfrecord'
casc_name = 'haarcascade_frontalface_alt.xml'
classifier_num = len(ckpt_name)
casc_path = os.path.join(data_folder_name, data_path_name, casc_name)
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
confusion_matrix = True
use_advanced_method = True
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)


def face_detect(image_path, casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_casccade_ = cv2.CascadeClassifier(casc_path_)
        img_ = cv2.imread(image_path)
        img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        # face detection
        faces = face_casccade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))


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


def advance_image(images_):
    rsz_img = []
    rsz_imgs = []
    for image_ in images_:
        rsz_img.append(image_)
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(image_[2:45, :], (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(cv2.flip(image_, 1), (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(image_[:, 2:45], (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(image_[2:45, 2:45], (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    return rsz_imgs


def produce_result(session_, t_x_in_, class_weight, sessions_, images_, x_in_, drop_, logit_):
    images_ = np.multiply(np.array(images_), 1./255)
    if use_advanced_method:
        rsz_imgs = advance_image(images_)
    else:
        rsz_imgs = [images_]
    pred_logits_ = []
    for rsz_img in rsz_imgs:
        pred_prob_mul = pred_proba_mul(rsz_img, sessions_, x_in_, drop_, logit_)
        t_x_reshape = tf.reshape(t_x_in_, [classifier_num, -1])
        model_output = tf.reshape(tf.matmul(class_weight, t_x_reshape), [-1, num_class])
        pred_logits_.append(session_.run(tf.nn.softmax(model_output), {t_x_in_: pred_prob_mul}))
    return np.sum(pred_logits_, axis=0)


def produce_results(session_, t_x_in_, class_weight, sessions_, images_, x_in_, drop_, logit_):
    results = []
    pred_logits_ = produce_result(session_, t_x_in_, class_weight, sessions_, images_, x_in_, drop_, logit_)
    pred_logits_list_ = np.array(np.reshape(np.argmax(pred_logits_, axis=1), [-1])).tolist()
    for num in range(num_class):
        results.append(pred_logits_list_.count(num))
    result_decimals = np.around(np.array(results) / len(images_), decimals=3)
    return results, result_decimals


def produce_confusion_matrix(session_, t_x_in_, class_weight, sessions_, images_list_,
                             x_in_, drop_, logit_, total_num_):
    total = []
    total_decimals = []
    for ii, images_ in enumerate(images_list_):
        results, result_decimals = produce_results(session_, t_x_in_, class_weight, sessions_,
                                                   images_, x_in_, drop_, logit_)
        total.append(results)
        total_decimals.append(result_decimals)
        print(results, ii, ":", result_decimals[ii])
        print(result_decimals)
    sum = 0
    for i_ in range(num_class):
        sum += total[i_][i_]
    print('acc: {:.3f} %'.format(sum * 100. / total_num_))
    print('Using ', ckpt_name)


def predict_emotion(session_, t_x_in_, class_weight, sessions_, image_, x_in_, drop_, logit_):
    image_ = cv2.resize(image_, (default_height, default_width))
    image_ = np.reshape(image_, [-1, default_height, default_width, channel])
    return produce_result(session_, t_x_in_, class_weight, sessions_, image_, x_in_, drop_, logit_)[0]


def main(argv):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        classifier_weight = tf.Variable([[1.0241427,  0.97796094, 0.9731235,  1.0292563,  1.0135012,  0.98942924]], dtype=tf.float32, name='classifier_weight')
        t_x_input = tf.placeholder(dtype=tf.float32, shape=[classifier_num, None, num_class])

        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver({"classifier_weight": classifier_weight})
        # saver.restore(sess, save_ckpt_path)
        # print(sess.run(classifier_weight))
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

        if not confusion_matrix:
            images_path = []
            files = os.listdir(pic_path)
            for file in files:
                if file.lower().endswith('jpg') or file.endswith('png'):
                    images_path.append(os.path.join(pic_path, file))
            for image in images_path:
                faces, img_gray, img = face_detect(image)
                spb = img.shape
                sp = img_gray.shape
                height = sp[0]
                width = sp[1]
                size = 600
                emotion_pre_dict = {}
                face_exists = 0
                for (x, y, w, h) in faces:
                    face_exists = 1
                    face_img_gray = img_gray[y:y + h, x:x + w]
                    results_sum = predict_emotion(sess, t_x_input, classifier_weight, sessions, face_img_gray, x_input,
                                                  dropout, logits)  # face_img_gray
                    for i, emotion_pre in enumerate(results_sum):
                        emotion_pre_dict[emotion_labels[i]] = emotion_pre
                    # 输出所有情绪的概率
                    print(emotion_pre_dict)
                    label = np.argmax(results_sum)
                    emo = emotion_labels[int(label)]
                    print('Emotion : ', emo)
                    # 输出最大概率的情绪
                    # 使框的大小适应各种像素的照片
                    t_size = 2
                    ww = int(spb[0] * t_size / 300)
                    www = int((w + 10) * t_size / 100)
                    www_s = int((w + 20) * t_size / 100) * 2 / 5
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
                    cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                www_s, (255, 0, 255), thickness=www, lineType=1)
                    # img_gray full face     face_img_gray part of face
                if face_exists:
                    cv2.namedWindow('Emotion_classifier', 0)
                    cent = int((height * 1.0 / width) * size)
                    cv2.resizeWindow('Emotion_classifier', size, cent)
                    cv2.imshow('Emotion_classifier', img)
                    k = cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    # if k & 0xFF == ord('q'):
                    #     break
        if confusion_matrix:
            with open(csv_path, 'r') as f:
                csvr = csv.reader(f)
                header = next(csvr)
                rows = [row for row in csvr]
                val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
                tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
            confusion_images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
            test_set = tst
            total_num = len(test_set)
            for label_image_ in test_set:
                label_ = int(label_image_[0])
                image_ = np.reshape(np.asarray([int(p) for p in label_image_[-1].split()]),
                                    [default_height, default_width, 1])
                confusion_images[label_].append(image_)
            produce_confusion_matrix(sess, t_x_input, classifier_weight, sessions, confusion_images.values(), x_input,
                                     dropout, logits, total_num)


if __name__ == '__main__':
    tf.app.run()
