import time
import os
import cv2
import numpy as np
from PIL import Image
import random
from utils.listflowfile import dataloader as get_data


def write_each_txt(file_list, txt):

    file_txt = open("model_data/"+txt, "w")

    for name in file_list:
        file_txt.write(name + "\n")

    file_txt.close()


if __name__ == "__main__":

    filepath = "/ssd2/xufudong_dataset/sceneflow/"
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = get_data(filepath)
    # print(len(all_left_img), len(all_right_img), len(all_left_disp)): 35454 35454 35454
    # print(len(test_left_img), len(test_right_img), len(test_left_disp)): 4370 4370 4370

    # 总共训练集有35454张图片，验证集有4370张图片
    train_data = []
    for k in range(len(all_left_disp)):

        img_l_path = all_left_img[k]
        img_r_path = all_right_img[k]
        disp_path = all_left_disp[k]

        combine = img_l_path + "#" + img_r_path + "#" + disp_path
        train_data.append(combine)

    index = list(np.arange(0, len(train_data), 1))

    # 打乱多次
    for k in range(1000):
        random.seed(k)
        random.shuffle(index)
    shuffle_data = [train_data[k] for k in index]

    # 划分labeled训练集和pseudo-labeled训练集
    k1 = int(0.125 * len(train_data))
    print("train: labeled & pseudo-labeled ", k1, len(train_data)-k1)

    train_file = shuffle_data[:k1]
    val_file = shuffle_data[k1:]
    write_each_txt(train_file, 'labeled.txt')
    write_each_txt(val_file, 'pseudo_labeled.txt')

    test_data = []
    for k in range(len(test_left_disp)):
        img_l_path = test_left_img[k]
        img_r_path = test_right_img[k]
        disp_path = test_left_disp[k]

        combine = img_l_path + "#" + img_r_path + "#" + disp_path
        test_data.append(combine)
    write_each_txt(test_data, 'test.txt')








