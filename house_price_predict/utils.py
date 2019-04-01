import numpy as np
import os
import re
import pickle
import pymysql


def cal_mean_std(city_p_dict: dict):
    price_list_ = list()
    for k, v in city_p_dict.items():
        if len(v) == 0:
            pass
        else:
            for v_item in v:
                if bool(re.search(r'^\d', v_item[1])):
                    price = float(re.match(r'^\D*(\d+\.*\d*)\D*', v_item[1]).group(1))
                    if len(str(price)) > 11:
                        continue
                    price_list_.append(price)
    return np.mean(price_list_), np.std(price_list_)


def build_map(city_p_dict: dict):
    city2num_map_ = dict()
    city2num_map_["默认"] = 0
    for k in city_p_dict:
        city2num_map_[k] = len(city2num_map_)
    return city2num_map_


def load_data():
    city_price_dict = dict()
    data_path = 'F:\Python\\anjuke_spider\\new_price_pkl'
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
    return city_price_dict


def norm_date(date_: float):
    return (date_-200000)/1000


def norm_price(price_: float):
    return np.log10(price_/1000)


def recover(list_):
    return np.multiply(np.power(10, list_), 1000)


def recover_date(list_):
    return np.add(np.multiply(list_, 1000), 200000)


def cal_loss(arr1, arr2, len_, city, city2num_map):
    i = 0
    count = 0
    bback = 0
    sum_ = np.array([0])
    sum_pre = np.array([0])
    num2city_map = dict(zip(city2num_map.values(), city2num_map.keys()))

    # 存储到数据库的部分
    to_sql_pred_now_price = []

    for a, b in zip(arr1, arr2):
        a_price = recover(a[len_[i]-1-bback])
        b_price = recover(b[len_[i]-1-bback])
        bias = np.abs(np.subtract(a_price, b_price))
        bias_pre = np.abs(np.subtract(recover(b[len_[i]-2]), recover(b[len_[i]-1])))
        # if bias > 6700:
        #     print(bias, num2city_map[city[i][0]])
        if bias < bias_pre:
            # print(bias, ": ", bias_pre, num2city_map[city[i][0]])
            count += 1
        sum_ = np.add(sum_, bias)
        sum_pre = np.add(sum_pre, bias_pre)
        to_sql_pred_now_price.append([num2city_map[city[i][0]], '2019', '03', round(a_price, 1)])
        i += 1

    print(i, count)
    print(sum_/i)
    print(sum_pre/i)
    return sum_ / i


def get_connection():
    connection = pymysql.Connect(host='42.159.122.43', user='root',
                                 password='123456', db='MBH', charset='utf8')
    pymysql.charset = 'utf-8'
    # conn = create_engine('mysql+mysqldb://root:123456@42.159.122.43:3306/MBH?charset=utf8')
    return connection


def house_list_save(res: list):
    connection = get_connection()
    cursor = connection.cursor()
    sql = 'insert into PredictPrice(location, year, month, predict_price) values(%s, %s, %s, %s)'
    args = res
    try:
        cursor.executemany(sql, args)
    except Exception as e:
        print('执行Mysql: % s时出错： % s' % (sql, e))
    finally:
        cursor.close()
        connection.commit()
        connection.close()
        print("all saved to sql")


def wash_dict(city_p_dict: dict, norm_flag=True):
    dict_items = {}
    for k, v in city_p_dict.items():
        if len(v) <= 50:
            pass
        else:
            # start flag
            try:
                dict_items[k].append([-1, -1])
            except KeyError:
                dict_items[k] = list()
                dict_items[k].append([-1, -1])
            for v_item in v:
                year = re.match(r'^\d{4}', v_item[0]).group(0)
                month = re.match(r'^\d{4}\D+(\d+\.*\d*)\D*', v_item[0]).group(1)
                if bool(re.search(r'^\d', v_item[1])):
                    price = float(re.match(r'^\D*(\d+\.*\d*)\D*', v_item[1]).group(1))

                    if len(str(price)) > 11:
                        continue
    #                 one_item = [k, year, month, price, change]
                    try:
                        if norm_flag:
                            dict_items[k].append([norm_date(float(year+month)), norm_price(price)])
                        else:
                            dict_items[k].append([float(year + month), price])
                    except KeyError:
                        dict_items[k] = list()
                        if norm_flag:
                            dict_items[k].append([norm_date(float(year+month)), norm_price(price)])
                        else:
                            dict_items[k].append([float(year + month), price])
#                 dict_items.append(one_item)
    for k, v in dict_items.items():
        dict_items[k] = sorted(dict_items[k], key=lambda x: x[0])
    return dict_items
