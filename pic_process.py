import numpy as np
import os
import cv2


pic_width = 100
pic_height = 100
train_set_hotdog = os.getcwd() + '/train/hotdog/'
train_set_dog = os.getcwd() + '/train/dog/'
train_set_resized = os.getcwd() + '/train/resized/'
test_set_hotdog = os.getcwd() + '/test/hotdog/'
test_set_dog = os.getcwd() + '/test/dog/'
test_set_resized = os.getcwd() + '/test/resized/'


def set_label(dir_A, dir_B, dir_resized):
    pic_path_list = []
    label = []
    resized_path_list = []

    # set labels
    for filename in os.listdir(dir_A):
        pic_path_list.append(dir_A + filename)
        label.append(0)

    for filename in os.listdir(dir_B):
        pic_path_list.append(dir_B + filename)
        label.append(1)

    pic_counter = 0
    for i in pic_path_list:
        print(pic_counter)
        pic = cv2.imread(i)
        resized_pic = cv2.resize(pic, (pic_width, pic_height))
        file_in_resized = dir_resized + str(pic_counter) + '.jpg'
        cv2.imwrite(file_in_resized, resized_pic)
        resized_path_list.append(file_in_resized)
        pic_counter += 1

    data_required = np.array([resized_path_list, label])
    data_required = data_required.transpose()
    np.random.shuffle(data_required)

    return data_required


def process_data(data):
    pic_matrix = []
    for i in data:
        pic = cv2.imread(i[0])
        pic = pic.astype(float)
        temp = pic.reshape((pic_width*pic_height, 3))
        pic_reshape = np.hstack((temp[:, 2], temp[:, 1], temp[:, 0]))
        pic_matrix.append(pic_reshape)

    pic_matrix = np.asarray(pic_matrix)
    return pic_matrix


def prepare_data():
    train_list = set_label(train_set_hotdog, train_set_dog, train_set_resized)
    test_list = set_label(test_set_hotdog, test_set_dog, test_set_resized)

    train_label = train_list[:, 1]
    test_label = test_list[:, 1]
    train_label = train_label.astype(int)
    test_label = test_label.astype(int)

    train_data = process_data(train_list)
    test_data = process_data(test_list)

    mean_pic = np.mean(train_data, axis=0)
    train_data -= mean_pic
    test_data -= mean_pic

    classes = ['objectA', 'objectB']

    data_dict = {
        'pic_train': train_data,
        'label_train': train_label,
        'images_test': test_data,
        'label_test': test_label,
        'classes': classes
    }
    return data_dict


def main():
    prepare_data()
