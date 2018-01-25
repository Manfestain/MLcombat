# _*_ coding:utf-8 _*_

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/reshape:0'

JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '/path/to/model'
MODEL_FILE = 'classify_image_graph_def.pb'

CACHE_DIR = '/tmp/bottleneck'

INPUT_DATA = '/path/to/flower_data'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 3000
BATCH = 100

def create_image_lists(testing_percentage, validation_percenatge):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]   # 遍历文件目录INPUT_DATA
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)   # 取主文件名
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_percenatge:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percenatge):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images.append,
            'validation': validation_images
        }

    return result

def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label


