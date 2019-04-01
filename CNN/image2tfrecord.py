import os
import tensorflow as tf

data_folder_name = '..\\temp'
data_path_name = 'cv'
pic_path_name = 'fer2013\\train'
record_name = 'fer2013_train.tfrecord'
channel = 1
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)
record_path = os.path.join(data_folder_name, data_path_name, record_name)


def pics_lists(pic_path_):
    pics_locations_ = [os.path.join(pic_path_, loc) for loc in os.listdir(pic_path_)
                       if os.path.isdir(os.path.join(pic_path_, loc))]
    pics_names_ = [os.listdir(pics_loc) for pics_loc in pics_locations_]
    pics_locs_ = []
    for pics_loc, pics_location_ in zip(pics_names_, pics_locations_):
        pics_loc_ = []
        for pic_loc in pics_loc:
            pics_loc_.append(os.path.join(pics_location_, pic_loc))
        pics_locs_.append(pics_loc_)
    print(pics_locs_)
    return pics_locs_


def write_binary(record_name_, pics_locs_):
    writer_ = tf.python_io.TFRecordWriter(record_name_)
    with tf.Session() as sess:
        for pics_label_, pics_loc_ in enumerate(pics_locs_):
            print("In:", pics_loc_)
            for pic_loc_ in pics_loc_:
                decode_jpg_data_ = tf.placeholder(tf.string)
                decode_jpg_ = tf.image.decode_jpeg(decode_jpg_data_, channels=channel)
                image_data_ = tf.gfile.GFile(pic_loc_, 'rb').read()
                image_ = sess.run(decode_jpg_, {decode_jpg_data_: image_data_})
                height_, width_ = image_.shape[0], image_.shape[1]
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image/label": tf.train.Feature(int64_list=tf.train.Int64List(value=[pics_label_])),
                            "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height_])),
                            "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width_])),
                            "image/raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data_]))
                        }
                    )
                )
                writer_.write(example.SerializeToString())
    writer_.close()


pics_locs = pics_lists(pic_path)
write_binary(record_path, pics_locs)

