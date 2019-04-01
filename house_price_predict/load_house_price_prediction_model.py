import os
import pickle
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import lstm_model_embed_house
from tqdm import tqdm
from utils import *
ckpt_path = './model/backup2/house_price_embed_pre2.ckpt'
city_price_dict = load_data()
pad_to_num = 112
city_length = 7118
retrain_flag = True
city2num_map = build_map(city_price_dict)


def wash_dict(city_p_dict: dict):
    city_year_month_price_dict = dict()
    for k, v in city_p_dict.items():
        if len(v) <= 3:
            pass
        else:
            # start flag
            try:
                city_year_month_price_dict[k].append([-1, -1])
            except KeyError:
                city_year_month_price_dict[k] = list()
                city_year_month_price_dict[k].append([-1, -1])
            for v_item in v:
                year = re.match(r'^\d{4}', v_item[0]).group(0)
                month = re.match(r'^\d{4}\D+(\d+\.*\d*)\D*', v_item[0]).group(1)
                if bool(re.search(r'^\d', v_item[1])):
                    price = float(re.match(r'^\D*(\d+\.*\d*)\D*', v_item[1]).group(1))

                    if len(str(price)) > 11:
                        continue
    #                 one_item = [k, year, month, price, change]
                    try:
                        city_year_month_price_dict[k].append([norm_date(float(year+month)), norm_price(price)])
                    except KeyError:
                        city_year_month_price_dict[k] = list()
                        city_year_month_price_dict[k].append([norm_date(float(year + month)), norm_price(price)])
    return city_year_month_price_dict


def split_data(data_dict: dict, city2num_map_: dict):
    x_dict_ = dict()
    y_dict_ = dict()
    house_dict = dict()
    sequence_length_ = dict()
    for k, v in data_dict.items():
        # -------------------
        x_dict_[k] = v[:-1]
        y_dict_[k] = [x[1] for x in v][1:-1]
        # -------------------
        sequence_length_[k] = len(x_dict_[k])
        house_dict[k] = [city2num_map_[k]]*len(x_dict_[k])
    return x_dict_, y_dict_, sequence_length_, house_dict


def get_city(city_name, x_, y_, seq_len_, house_dict_):
    return np.array([x_[city_name]]), np.array([y_[city_name]]), \
           np.array([seq_len_[city_name]]), np.array([house_dict_[city_name]])


def generate_data(pred_, x_, seq_len_, city_in):
    if len(pred_) == 0:
        return x_, seq_len_, city_in
    else:
        date_now = int(np.round(x_[0][-1][0]*1000+200000))
        if str(date_now)[-2:] != "12":
            date_now = date_now + 1.
        else:
            date_now = float("{}{}".format(int(date_now/100)+1, "01"))
        # print(date_now)
        pred_price = pred_[0][seq_len_[0]-1]
        add_x_b = np.reshape([norm_date(date_now), pred_price], [1, -1, 2])
        x_b = np.append(x_, add_x_b, axis=1)
        seq_len_b = [seq_len_[0]+1]
        city_b = np.append(city_in[0], [city_in[0][0]], axis=0)
        city_b = np.reshape(city_b, [1, -1])
        return x_b, seq_len_b, city_b


city_year_month_price_dict = wash_dict(city_price_dict)
for k, v in city_year_month_price_dict.items():
    city_year_month_price_dict[k] = sorted(city_year_month_price_dict[k], key=lambda x: x[0])

x_, y_, seq_len_, house_dict_ = split_data(city_year_month_price_dict, city2num_map)

sess = tf.Session()
saver = tf.train.import_meta_graph(ckpt_path+'.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()

x_in = graph.get_tensor_by_name("x_input:0")
y_target = graph.get_tensor_by_name("y_target:0")
seq_len_in = graph.get_tensor_by_name("sequence_length:0")
logits_pred = graph.get_tensor_by_name("project/output/logits:0")
city_in = graph.get_tensor_by_name("city_input:0")
drop_out = graph.get_tensor_by_name("dropout:0")

res = []
for index, city in tqdm(enumerate(city2num_map)):
    if index == 0:
        continue
    # city = "白石桥"
    try:
        x_batch, y_batch, seq_len_batch, city_batch = get_city(city, x_, y_, seq_len_, house_dict_)
    except KeyError:
        continue
    if seq_len_batch[0] < 50:
        continue
    pred = []
    # -------------------
    pre_month = 1
    # -------------------
    for i in range(pre_month):
        x_batch, seq_len_batch, city_batch = generate_data(pred, x_batch, seq_len_batch, city_batch)
        feed_dict_ = {x_in: x_batch, y_target: [[0]*seq_len_batch[0]], seq_len_in: seq_len_batch,
                      city_in: city_batch, drop_out: 1}
        pred = sess.run(logits_pred, feed_dict_)
        price_now = round(float(recover(pred[0][seq_len_batch[0]-1])), 1)
        city_now = city
        year_now = "2019"
        # month_now = "0{}".format(4 + i)
        month_now = "0{}".format(3 + i)
        res.append([city_now, year_now, month_now, price_now])
        # pass
        # print("predict: {}\ntarget: {}".format(recover(pred[0][:seq_len_batch[0]]),
        #                                        recover(y_batch[0][:seq_len_batch[0]])))
print(len(res))
house_list_save(res)
# plt.plot(range(seq_len_batch[0])[2:], recover(pred[0][2:seq_len_batch[0]]), label='predict')
# plt.plot(range(seq_len_batch[0]-pre_month)[2:], recover(y_batch[0][2:seq_len_batch[0]-pre_month+1]), label='target')
# plt.legend()
# plt.show()
