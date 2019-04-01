import tensorflow as tf
import numpy as np
from Bi_LSTM_CRF import bilstm
from tensorflow.python.framework import ops
import os
import helper

data_folder_name = 'temp'
data_path_name = 'cn_nlp'
data_name = '199801out.txt'
data_name2 = '2014out.txt'
ckpt_name = 'cbow_movie_embeddings.ckpt'
save_ckpt_name = 'bilstm_crf_cn.ckpt'
vocab_name = 'bilstm_crf_cn.pkl'

batch_size = 512
test_batch_size = 2048
lstm_dim = 100
embedding_size = 200
generations = 1200
num_tags = 4
save_flag = False

ops.reset_default_graph()
sess = tf.Session()

if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

# model_checkpoint_path = os.path.join(data_folder_name, ckpt_name)
# with open(os.path.join(data_folder_name, vocab_name), 'rb') as f:
#     word_dict = pickle.load(f)
#
# embeddings = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=0.1))
# saver = tf.train.Saver({"embeddings": embeddings})
# saver.restore(sess, model_checkpoint_path)


def viterbi_decode(score_, trans_):
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(score_, trans_)
    return viterbi_sequence


def evaluate(scores_, lengths_, trans_, targets_):
    correct_seq = 0.
    total_len = 0.
    for ix, score_ in enumerate(scores_):
        score_real = score_[:lengths_[ix]]
        target_real = targets_[ix][:lengths_[ix]]
        pre_sequence = viterbi_decode(score_real, trans_)
        correct_seq += np.sum((np.equal(pre_sequence, target_real)))
        total_len += lengths_[ix]
    return correct_seq/total_len


texts, targets, lengths = helper.load_data(data_name, data_name2)
max_len = max(lengths)
print('max_len:', max_len)
word_dict = helper.build_dictionary(texts, list=True)
# with open(os.path.join(data_folder_name, data_path_name, vocab_name), 'wb') as f:
#     pickle.dump(word_dict, f)
# print(len(word_dict))
# exit()
texts = helper.text_to_numbers(texts, word_dict, list=True)
normalized_texts = np.array([(x+[0]*max_len)[0:max_len] for x in texts])
normalized_targets = np.array([(x+[0]*max_len)[0:max_len] for x in targets])

train_indices = np.random.choice(len(targets), round(0.9 * len(targets)), replace=False)
test_indices = np.sort(np.array(list(set(range(len(targets))) - set(train_indices))))
texts_train = normalized_texts[train_indices]
texts_test = normalized_texts[test_indices]
target_train = normalized_targets[train_indices]
target_test = normalized_targets[test_indices]
lengths_train = lengths[train_indices]
lengths_test = lengths[test_indices]

vocabulary_size = len(word_dict)

embeddings = tf.Variable(tf.truncated_normal(
    [vocabulary_size, embedding_size], stddev=0.05))
bilstm_model = bilstm.BiLSTM_Model(embeddings)

logits = bilstm_model.logits
loss = bilstm_model.loss
train_step = bilstm_model.train_step

init = tf.global_variables_initializer()
sess.run(init)
print('start training')
saver = tf.train.Saver(max_to_keep=1)
max_accuracy = 0
for i in range(generations):
    rand_indices = np.random.choice(len(target_train), batch_size)
    x_batch = texts_train[rand_indices]
    y_batch = target_train[rand_indices]
    batch_len = lengths_train[rand_indices]

    feed_dict = {bilstm_model.x_input: x_batch, bilstm_model.y_target: y_batch,
                 bilstm_model.dropout: 0.5, bilstm_model.max_steps: batch_len}
    sess.run(train_step, feed_dict)
    train_loss, train_logits, trans_martix = sess.run([loss, logits, bilstm_model.trans], feed_dict)
    train_accuracy = evaluate(train_logits, batch_len, trans_martix, y_batch)
    print('Generation # {}. Train Loss : {:.3f} . '
          'Train Acc : {:.3f}'.format(i, train_loss, train_accuracy))
    if (i+1) % 10 == 0:
        test_rand_indices = np.random.choice(len(target_test), test_batch_size)
        test_x_batch = texts_test[test_rand_indices]
        test_y_batch = target_test[test_rand_indices]
        test_batch_len = lengths_test[test_rand_indices]
        test_feed_dict = {bilstm_model.x_input: test_x_batch, bilstm_model.y_target: test_y_batch,
                          bilstm_model.dropout: 0, bilstm_model.max_steps: test_batch_len}

        test_loss, test_logits = sess.run([loss, logits], test_feed_dict)
        test_accuracy = evaluate(test_logits, test_batch_len, trans_martix, test_y_batch)
        print('Generation # {}. Test Loss : {:.3f} . '
              'Test Acc : {:.3f}'.format(i, test_loss, test_accuracy))
        if test_accuracy >= max_accuracy and save_flag:
            max_accuracy = test_accuracy
            saver.save(sess, os.path.join(data_folder_name, data_path_name, save_ckpt_name))
            print('Generation # {}. --model saved--'.format(i))
print('acc:', max_accuracy)



