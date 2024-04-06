import time
import os
import cv2
import numpy as np
from PIL import Image


if __name__ == "__main__":

    with open("model_data/semi_stage1.txt") as file:
        lines = file.readlines()

    for k in range(100):
        line = lines[k]
        obj1 = line.split('\n')  # 去除换行符
        obj2 = obj1[0].split("#")

        left_path = obj2[0]
        right_path = obj2[1]
        disp_path = obj2[2]

        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        disp_L = np.load(disp_path)

        # cv2.imshow("img_opencv_left", img_opencv_left)
        # cv2.imshow("img_opencv_right", img_opencv_right)
        # cv2.imshow("dataL", disp_L/np.max(disp_L))
        # cv2.waitKey(10)

        directory = "demo3/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        cv2.imwrite("demo3/img_" + str(k).zfill(4) + ".png", img_left / np.max(img_left) * 255)
        cv2.imwrite("demo3/disp_" + str(k).zfill(4) + ".png", disp_L / np.max(disp_L) * 255)

        disp_L[disp_L > 84] = 84
        heatmap = cv2.applyColorMap(np.uint8(255 * (disp_L / 84)), cv2.COLORMAP_JET)
        heatmap[np.where(heatmap < 0)] = 0
        cv2.imwrite("demo3/heatmap_" + str(k).zfill(4) + ".png", heatmap / np.max(heatmap) * 255)





