from PIL import Image
import random
import cv2
import os
import numpy as np


def main():
    with open("model_data/pseudo_labeled.txt") as file:
        lines = file.readlines()

    path_list = []
    for k in range(len(lines)):
        line = lines[k]
        obj1 = line.split('\n')  # 去除换行符
        obj2 = obj1[0].split("#")

        left_path = obj2[0]
        right_path = obj2[1]
        
        obj = left_path.split("/")
        name = "/".join(obj[3:-1]) + "/"
        img_name = obj[-1]
        disp_path = "Pseudo_stage1/" + name + img_name[:-3] + "npy"
        if not os.path.exists(disp_path):
            continue

        combine = left_path + "#" + right_path + "#" + disp_path
        path_list.append(combine)

    print(len(path_list))
    file_txt = open("model_data/semi_stage1.txt", "w")
    for name in path_list:
        file_txt.write(name + "\n")
    file_txt.close()


if __name__ == "__main__":
    main()





