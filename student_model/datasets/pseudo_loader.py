import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision
from copy import deepcopy

from datasets.data_io import get_transform, read_all_lines, pfm_imread


class SceneFlowDatset(Dataset):
    def __init__(self, list_filename1, list_filename2, training):
        self.left_filenames1, self.right_filenames1, self.disp_filenames1 = self.load_path(list_filename1)
        self.left_filenames2, self.right_filenames2, self.disp_filenames2 = self.load_path(list_filename2)
        print("list_filename1", len(self.left_filenames1), len(self.right_filenames1), len(self.disp_filenames1))
        print("list_filename2", len(self.left_filenames2), len(self.right_filenames2), len(self.disp_filenames2))

        # 维持真标签和伪标签平衡
        self.left_filenames_o = self.left_filenames1 * 8 + self.left_filenames2
        self.right_filenames_o = self.right_filenames1 * 8 + self.right_filenames2
        self.disp_filenames_o = self.disp_filenames1 * 8 + self.disp_filenames2
        print("Total", len(self.left_filenames_o), len(self.right_filenames_o), len(self.disp_filenames_o))

        # 打乱顺序，打乱多次
        index = list(np.arange(0, len(self.left_filenames_o), 1))
        for k in range(100):
            random.seed(k)
            random.shuffle(index)
        self.left_filenames = [self.left_filenames_o[k] for k in index]
        self.right_filenames = [self.right_filenames_o[k] for k in index]
        self.disp_filenames = [self.disp_filenames_o[k] for k in index]

        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split("#") for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]

        left_images = left_images[:100]
        right_images = right_images[:100]
        disp_images = disp_images[:100]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        left_path = self.left_filenames[index]
        right_path = self.right_filenames[index]
        disp_path = self.disp_filenames[index]

        obj = disp_path[-3:]
        if obj != "npy":    # 真实标签
            left_img = self.load_image(left_path)
            right_img = self.load_image(right_path)
            disparity = self.load_disp(disp_path)
        else:    # 伪标签
            left_img = self.load_image(left_path)
            right_img = self.load_image(right_path)
            disparity = np.load(disp_path)

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            if random.randint(0, 10) >= int(8):
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.3 * h), h - crop_h)    # 尽量在图像下方裁减，画面信息量更多

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # 存储原先rgb无损图像，用于专门计算重建rgb损失
            raw_left_img = deepcopy(left_img)
            raw_right_img = deepcopy(right_img)

            # 对左右两幅图像，分别施加亮度、gamma、对比度增强
            random_brightness = np.random.uniform(0.5, 2.0, 2)
            random_gamma = np.random.uniform(0.8, 1.2, 2)
            random_contrast = np.random.uniform(0.8, 1.2, 2)
            # print(random_brightness, random_gamma, random_contrast)： [1.2336095] [1.04901289] [0.95220439]

            left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
            left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
            left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
            right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
            right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
            right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
            right_img = np.asarray(right_img)
            left_img = np.asarray(left_img)

            # 以0.2的概率对右目图像进行部分区域随机缺失（用平均像素值代替各处像素值），相当于增加了网络学习的鲁棒性
            # 我在这里的SDA处做了修改，不光rgb图像有遮挡，我还将视差真值遮挡部分的值改为0，这样可以避免之前给出的监督信息是错误的这种情况
            # 并且我将概率0.2调为了0.5
            left_img = np.array(left_img)
            if np.random.binomial(1, 0.5):
                sx = int(np.random.uniform(35, 100))
                sy = int(np.random.uniform(25, 75))
                cx = int(np.random.uniform(sx, right_img.shape[0] - sx))
                cy = int(np.random.uniform(sy, right_img.shape[1] - sy))
                left_img[cx - sx:cx + sx, cy - sy:cy + sy] = np.mean(np.mean(left_img, 0), 0)[np.newaxis, np.newaxis]
                disparity[cx - sx:cx + sx, cy - sy:cy + sy] = 0

            """
            cv2.imshow("left", cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
            cv2.imshow("right", cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
            cv2.imshow("disp", disparity / np.max(disparity))
            cv2.waitKey(0)
            """

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            raw_left_img = processed(raw_left_img)
            raw_right_img = processed(raw_right_img)

            return {"raw_left": raw_left_img,
                    "raw_right": raw_right_img,
                    "left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "left_name": self.left_filenames[index],
                    "disp_name": self.disp_filenames[index]}

        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0}
