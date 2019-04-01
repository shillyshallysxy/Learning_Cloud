import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import lstm_model
from utils import *
data_path = 'F:\Python\\anjuke_spider\\new_price_pkl'
save_path = './model/house_price_3.ckpt'
city_price_dict = dict()

for father_path, dir_list, file_list in os.walk(data_path):
    for file_name in file_list:
        with open(os.path.join(father_path, file_name), 'rb') as f:
            temp_dict = pickle.load(f)
            for k, v in temp_dict.items():
                try:
                    city_price_dict[k].extend(v)
                except KeyError:
                    city_price_dict[k] = list()
                    city_price_dict[k].extend(v)

embedding_size = 20
city_year_month_price_dict = dict()
batch_size = 128
generations = 500
retrain_flag = True


def norm_date(date_: float):
    return np.log10(date_)


def norm_price(price_: float):
    return np.log10(price_)


def wash_dict(city_p_dict: dict, city2num_map_: dict):
    dict_items = []
    for k, v in city_p_dict.items():
        if len(v) == 0:
            pass
        else:
            for v_item in v:
                year = re.match(r'^\d{4}', v_item[0]).group(0)
                month = re.match(r'^\d{4}\D+(\d+\.*\d*)\D*', v_item[0]).group(1)
                if bool(re.search(r'^\d', v_item[1])):
                    price = float(re.match(r'^\D*(\d+\.*\d*)\D*', v_item[1]).group(1))

                    if len(str(price)) > 11:
                        continue
    #                 one_item = [k, year, month, price, change]
                    try:
                        city_year_month_price_dict[k].append([norm_date(float(year+month)), norm_price(price),
                                                              np.log10(city2num_map_[k])])
                    except KeyError:
                        city_year_month_price_dict[k] = list()
                        city_year_month_price_dict[k].append([norm_date(float(year + month)), norm_price(price),
                                                              np.log10(city2num_map_[k])])

#                 dict_items.append(one_item)
    return dict_items


def split_x_y(data_dict: dict):
    x_dict_ = dict()
    y_dict_ = dict()
    sequence_length = dict()
    for k, v in data_dict.items():
        x_dict_[k] = v[0:-2]
        y_dict_[k] = [x[1] for x in v][1:-1]
    for k, v in x_dict_.items():
        append_num = 111 - len(v)
        x_dict_[k].extend([[0., 0., -1]]*append_num)
    for k, v in y_dict_.items():
        append_num = 111 - len(v)
        sequence_length[k] = len(v)
        y_dict_[k].extend([0.]*append_num)
    return x_dict_, y_dict_, sequence_length


def split_x_y_test(data_dict: dict):
    x_dict_ = dict()
    y_dict_ = dict()
    sequence_length = dict()
    for k, v in data_dict.items():
        x_dict_[k] = v[0:-1]
        y_dict_[k] = [x[1] for x in v][1:]
    for k, v in x_dict_.items():
        append_num = 111 - len(v)
        x_dict_[k].extend([[0., 0., -1]]*append_num)
    for k, v in y_dict_.items():
        append_num = 111 - len(v)
        sequence_length[k] = len(v)
        y_dict_[k].extend([0.]*append_num)
    return x_dict_, y_dict_, sequence_length


# 9*12+3
def generate_data(data_dict_x: dict, data_dict_y: dict, sequence_length: dict):
    data_list_x = list(data_dict_x.values())
    data_list_y = list(data_dict_y.values())
    sequence_length = list(sequence_length.values())
    random_indices = np.random.choice(len(data_list_x), batch_size, replace=False)
    x_ = np.array(data_list_x)[random_indices]
    y_ = np.array(data_list_y)[random_indices]
    seq_len_ = np.array(sequence_length)[random_indices]
    # for index_ in random_indices:
    #     append_num = 111-len(data_list[index_])
    #     data_list[index_].extend([0., 0., -1]*append_num)
    return x_, y_, seq_len_


def recover(list_):
    # return np.add(np.multiply(list_, price_std), price_mean)
    return np.power(10, list_)


price_mean, price_std = cal_mean_std(city_price_dict)
city2num_map = build_map(city_price_dict)
wash_dict(city_price_dict, city2num_map)

for k, v in city_year_month_price_dict.items():
    city_year_month_price_dict[k] = sorted(city_year_month_price_dict[k], key=lambda x: x[0])

city_length = len(city_year_month_price_dict)+1
print("city_length:", city_length)
# embedding = tf.truncated_normal([city_length, embedding_size], mean=0, stddev=0.5, dtype=tf.float32)

x_dict, y_dict, sequence_length = split_x_y(city_year_month_price_dict)

x_dict_test, y_dict_test, sequence_length_test = split_x_y(city_year_month_price_dict)

model = lstm_model.LSTM_Model(input_dim=3, lstm_dim_=50, hidden_dim_=25)
x_in = model.x_input
y_target = model.y_target
seq_len_in = model.sequence_length
logits_pred = model.logits
loss = model.loss
train_step = model.train_step
drop_out = model.dropout
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=1)
sess.run(tf.global_variables_initializer())
if retrain_flag:
    saver_load = tf.train.Saver()
    saver_load.restore(sess, save_path)
    print("re train!")
test_pred_seq = []
test_target_seq = []
print('start!')
print(price_mean, price_std)
for i in range(generations):
    x_batch, y_batch, seq_len_batch = generate_data(x_dict, y_dict, sequence_length)
    feed_dict_train = {x_in: x_batch, y_target: y_batch, seq_len_in: seq_len_batch, drop_out: 1}
    sess.run(train_step, feed_dict_train)
    if (i+1) % 20 == 0:
        train_pred, train_loss = sess.run([logits_pred, loss], feed_dict_train)
        print("train generation: {}\ntrain predict: {}\ntrain target: {}\ntrain loss: {}".
              format(i+1,
                     recover(train_pred[0][:seq_len_batch[0]]),
                     recover(y_batch[0][:seq_len_batch[0]]),
                     train_loss))
    if (i+1) % 40 == 0 and i > generations/2:
        x_batch_test, y_batch_test, seq_len_batch_test = generate_data(x_dict_test, y_dict_test, sequence_length_test)
        feed_dict_test = {x_in: x_batch_test, y_target: y_batch_test, seq_len_in: seq_len_batch_test, drop_out: 1}
        test_pred, test_loss = sess.run([logits_pred, loss], feed_dict_test)
        print("test generation: {}\ntest predict: {}\ntest target: {}\ntest loss: {}".
              format(i+1,
                     recover(test_pred[0][:seq_len_batch_test[0]]),
                     recover(y_batch_test[0][:seq_len_batch_test[0]]),
                     test_loss))
        plt.plot(range(seq_len_batch_test[0]), recover(test_pred[0][:seq_len_batch_test[0]]), label='predict')
        plt.plot(range(seq_len_batch_test[0]), recover(y_batch_test[0][:seq_len_batch_test[0]]), label='target')
        plt.legend()
        plt.show()
saver.save(sess, save_path)
print("saved !")
