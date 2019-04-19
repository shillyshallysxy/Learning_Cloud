import os
import jsonlines
import tensorflow as tf
from tqdm import tqdm
from SNLI_loader.text_helper import *
import pickle
from Transformer.transformer import *
data_folder_name = '..\\temp'
data_path_name = 'cn_nlp\\corpus\\SNLI\\snli_1.0'
test_snli_name = 'snli_1.0_test.jsonl'
vocab_name_en = 'translate_en_50.pkl'

vocab_path_en = os.path.join(data_folder_name, "cn_nlp\\translate", vocab_name_en)

with open(vocab_path_en, 'rb') as f:
    word_dict_en = pickle.load(f)


class ConfigModel(object):
    vocab_size_en = len(word_dict_en)
    channels = 400
    learning_rate = 0.001
    layer_num = 4
    is_training = True
    shuffle_pool_size = 2560
    dropout_rate = 0.1
    num_heads = 8
    batch_size = 128
    max_length = 100
    num_tags = 3


pm = ConfigModel()

data_path = os.path.join(data_folder_name, data_path_name)
label_to_num_dict = {'entailment': 0, "neutral": 1, 'contradiction': 2}
sentence_1 = list()
sentence_2 = list()
label = list()
with open(os.path.join(data_path, test_snli_name), 'rb') as f:
    for item in jsonlines.Reader(f):
        if len(normalize_text(item["sentence1"])) > pm.max_length or len(normalize_text(item["sentence2"])) > pm.max_length:
            continue
        try:
            label.append(label_to_num_dict[item["gold_label"]])
        except KeyError:
            continue
        sentence_1.append(normalize_text(item["sentence1"]))
        sentence_2.append(normalize_text(item["sentence2"]))
en_data_num_1 = text_to_numbers(sentence_1, word_dict_en)
en_data_num_2 = text_to_numbers(sentence_2, word_dict_en)

for i in range(len(en_data_num_1)):
    en_data_num_1[i] = (en_data_num_1[i]+[0]*pm.max_length)[:pm.max_length]

    en_data_num_2[i] = (en_data_num_2[i]+[0]*pm.max_length)[:pm.max_length]


def generate_data():
    rand_indices = np.random.choice(len(en_data_num_1), pm.batch_size, replace=False)
    data_1 = np.array(en_data_num_1)[rand_indices]
    data_2 = np.array(en_data_num_2)[rand_indices]
    data_label = np.array(label)[rand_indices]
    return data_1, data_2, data_label


if __name__ == '__main__':
    with tf.Session() as sess:
        input1 = tf.placeholder(tf.int32, [None, pm.max_length])
        input2 = tf.placeholder(tf.int32, [None, pm.max_length])
        target = tf.placeholder(tf.int32, [None, ])
        tsl = TransformerSNLI(input1, input2, target, pm)
        for i in range(1000):
            d1, d2, lb = generate_data()
            train_feed = {input1: d1, input2: d2, target: lb}
            sess.run(tsl.train_op, train_feed)
            if (i+1) % 50 == 0:
                sess.run(tsl.preds, train_feed)










