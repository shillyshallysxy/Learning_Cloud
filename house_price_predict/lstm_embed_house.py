import tensorflow as tf
import matplotlib.pyplot as plt
import lstm_model_embed_house
from utils import *
save_path = './model/backup2/house_price_embed_pre2.ckpt'
city_price_dict = load_data()
tf.reset_default_graph()


embedding_size = 80
lstm_dim = 100
batch_size = 64
generations = 2000
pad_to_num = 112
retrain_flag = True
pre_train_flag = False
load_pre_train_flag = False
random_default_city_flag = False
price_mean, price_std = cal_mean_std(city_price_dict)
city2num_map = build_map(city_price_dict)


def split_x_y(data_dict: dict, city2num_map_: dict):
    x_dict_ = dict()
    y_dict_ = dict()
    house_dict = dict()
    sequence_length_ = dict()
    # 是否进行预训练
    if pre_train_flag:
        print("预训练embeddings")
        for k, v in data_dict.items():
            bias = np.random.randint(1, 2)
            x_dict_[k] = v[0:bias]
            y_dict_[k] = [x[1] for x in v][1:bias+1]
            sequence_length_[k] = len(x_dict_[k])
            house_dict[k] = [city2num_map_[k]] * pad_to_num
        for k, v in x_dict_.items():
            append_num = pad_to_num - len(v)
            x_dict_[k].extend([[0., 0.]] * append_num)
        for k, v in y_dict_.items():
            append_num = pad_to_num - len(v)
            y_dict_[k].extend([0.] * append_num)
    else:
        bias_ = 2
        for k, v in data_dict.items():
            x_dict_[k] = v[0:-bias_]
            if (-bias_ + 1) == 0:
                y_dict_[k] = [x[1] for x in v][1:]
            else:
                y_dict_[k] = [x[1] for x in v][1:-bias_ + 1]
            sequence_length_[k] = len(x_dict_[k])
            if random_default_city_flag:
                if np.random.randint(1, 4) == 1:
                    # print("1111")
                    house_dict[k] = [0] * pad_to_num
                else:
                    house_dict[k] = [city2num_map_[k]] * pad_to_num
            else:
                house_dict[k] = [city2num_map_[k]] * pad_to_num
        for k, v in x_dict_.items():
            append_num = pad_to_num - len(v)
            x_dict_[k].extend([[0., 0.]] * append_num)
        for k, v in y_dict_.items():
            append_num = pad_to_num - len(v)
            y_dict_[k].extend([0.] * append_num)

    return x_dict_, y_dict_, sequence_length_, house_dict


def split_x_y_test(data_dict: dict, city2num_map_: dict):
    x_dict_ = dict()
    y_dict_ = dict()
    house_dict = dict()
    sequence_length_ = dict()
    bias_ = 1
    for k, v in data_dict.items():
        x_dict_[k] = v[0:-bias_]
        if (-bias_+1) == 0:
            y_dict_[k] = [x[1] for x in v][1:]
        else:
            y_dict_[k] = [x[1] for x in v][1:-bias_+1]
        sequence_length_[k] = len(x_dict_[k])
        house_dict[k] = [[city2num_map_[k]] * pad_to_num]
    for k, v in x_dict_.items():
        append_num = pad_to_num - len(v)
        x_dict_[k].extend([[0., 0.]] * append_num)
    for k, v in y_dict_.items():
        append_num = pad_to_num - len(v)
        y_dict_[k].extend([0.] * append_num)
    return x_dict_, y_dict_, sequence_length_, house_dict


# 9*12+3
def generate_data(data_dict_x: dict, data_dict_y: dict, sequence_length_: dict, house_dict_: dict,
                  batch_size_=batch_size):
    data_list_x = list(data_dict_x.values())
    data_list_y = list(data_dict_y.values())
    house_list_ = list(house_dict_.values())
    sequence_length_ = list(sequence_length_.values())
    random_indices = np.random.choice(len(data_list_x), batch_size_, replace=True)
    # x_ = np.array(data_list_x)[random_indices]
    # y_ = np.array(data_list_y)[random_indices]
    # seq_len_ = np.array(sequence_length_)[random_indices]
    # hs_ls_ = np.array(house_list_)[random_indices]
    x_ = np.array(data_list_x)
    y_ = np.array(data_list_y)
    seq_len_ = np.array(sequence_length_)
    hs_ls_ = np.array(house_list_)
    return x_, y_, seq_len_, hs_ls_


city_year_month_price_dict = wash_dict(city_price_dict)

city_length = len(city2num_map)
print("city_length:", city_length)
# embedding = tf.truncated_normal([city_length, embedding_size], mean=0, stddev=0.5, dtype=tf.float32)

x_dict, y_dict, sequence_length, city_num_list = split_x_y(city_year_month_price_dict, city2num_map)

x_dict_test, y_dict_test, sequence_length_test, city_num_list_test = split_x_y_test(city_year_month_price_dict, city2num_map)

model = lstm_model_embed_house.LSTM_Model(2, city_length, embedding_size, lstm_dim_=lstm_dim)
x_in = model.x_input
y_target = model.y_target
seq_len_in = model.sequence_length
logits_pred = model.logits
loss = model.loss
train_step = model.train_step
city_in = model.city_input
drop_out = model.dropout
sess = tf.Session()
sess.run(tf.global_variables_initializer())
if retrain_flag:
    saver_load = tf.train.Saver()
    saver_load.restore(sess, save_path)
    print("re train !")
# if not pre_train_flag:
if load_pre_train_flag:
    saver_load = tf.train.Saver({"embeddings": model.embeddings, "city_w": model.city_var, "city_b": model.city_bias})
    saver_load.restore(sess, "./model/pre_train.ckpt")
    print("get pre train embeddings !")
test_pred_seq = []
test_target_seq = []
print('start!')
print(price_mean, price_std)
for i in range(generations):
    x_batch, y_batch, seq_len_batch, city_batch = generate_data(x_dict, y_dict, sequence_length, city_num_list)
    feed_dict_train = {x_in: x_batch,
                       y_target: y_batch,
                       seq_len_in: seq_len_batch,
                       city_in: city_batch,
                       drop_out: 0.8}
    # sess.run(train_step, feed_dict_train)
    if (i+1) % 20 == 0:
        train_pred, train_loss = sess.run([logits_pred, loss], feed_dict_train)
        # print("train generation: {}\ntrain predict: {}\ntrain target: {}\ntrain loss: {}".
        print("train generation: {}\ntrain loss: {}".
              format(i+1,
                     # recover(train_pred[0][:seq_len_batch[0]]),
                     # recover(y_batch[0][:seq_len_batch[0]]),
                     train_loss))
    # if (i+1) % 100 == 0 and i > generations*2/3:
    if True:
        x_batch_test, y_batch_test, seq_len_batch_test, city_batch_test = \
            generate_data(x_dict_test, y_dict_test, sequence_length_test, city_num_list)

        feed_dict_test = {x_in: x_batch_test,
                          y_target: y_batch_test,
                          seq_len_in: seq_len_batch_test,
                          drop_out: 1,
                          city_in: city_batch_test}
        test_pred, test_loss = sess.run([logits_pred, loss], feed_dict_test)
        cal_loss(test_pred, y_batch_test, seq_len_batch_test, city_batch_test, city2num_map)
        print("test generation: {}\ntest predict: {}\ntest target: {}\ntest loss: {}".
              format(i+1,
                     recover(test_pred[0][:seq_len_batch_test[0]]),
                     recover(y_batch_test[0][:seq_len_batch_test[0]]),
                     test_loss))
        # plt.plot(range(len(test_pred[0][2:seq_len_batch_test[0]])), recover(test_pred[0][2:seq_len_batch_test[0]]), label='predict')
        # plt.plot(range(len(y_batch_test[0][2:seq_len_batch_test[0]])), recover(y_batch_test[0][2:seq_len_batch_test[0]]), label='target')
        # plt.legend()
        # plt.show()
if pre_train_flag:
    saver = tf.train.Saver({"embeddings": model.embeddings, "city_w": model.city_var, "city_b": model.city_bias})
    saver.save(sess, "./model/pre_train.ckpt")
    print("saved pre_train!")
else:
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(sess, save_path)
    print("saved model!")
