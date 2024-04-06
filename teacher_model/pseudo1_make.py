import time
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from datasets.data_io import pfm_imread
from models import __models__, model_loss
from utils import *
import datasets.data_io

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = np.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt > 0.05)
    return np.mean(err_mask)


def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = np.abs(D_gt - D_est)
    err_mask = E > thres
    return np.mean(err_mask)


def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return np.mean(np.abs(D_est - D_gt))


def main():
    with open("model_data/pseudo_labeled.txt") as file:
        lines = file.readlines()

    max_disp = 192
    model = __models__["gwcnet-gc"](max_disp)

    weight_path = "0098_3.59319_0.32577_0.96966_GwcNet_sceneflow.tar"
    state_dict = torch.load(weight_path)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict['state_dict'].items()})

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = nn.DataParallel(model)
        model.cuda()

    model.eval()
    calc_avg = []
    for k in range(len(lines)):

        line = lines[k]
        obj1 = line.split('\n')    # 去除换行符
        obj2 = obj1[0].split("#")

        left_path = obj2[0]
        right_path = obj2[1]
        disp_path = obj2[2]

        img_left = Image.open(left_path).convert('RGB')
        img_right = Image.open(right_path).convert('RGB')
        true_disp, scaleL = pfm_imread(disp_path)
        true_disp = np.ascontiguousarray(true_disp, dtype=np.float32)

        t11 = time.time()
        processed = datasets.data_io.get_transform()
        imgL = processed(img_left)
        imgR = processed(img_right)

        # pad to width and height to 16 times
        if imgL.shape[1] % 32 != 0:
            times = imgL.shape[1] // 32
            top_pad = (times + 1) * 32 - imgL.shape[1]
        else:
            top_pad = 0
        if imgL.shape[2] % 32 != 0:
            times = imgL.shape[2] // 32
            right_pad = (times + 1) * 32 - imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)
        if use_cuda:
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        # 表明当前计算不需要反向传播
        with torch.no_grad():
            [disp], [cost_softmax] = model(imgL, imgR)

        disp = torch.squeeze(disp)
        cost_softmax = torch.squeeze(cost_softmax)

        predict_disp = disp.data.cpu().numpy()
        predict_cost_softmax = cost_softmax.data.cpu().numpy()
        if top_pad != 0 and right_pad != 0:
            predict_disp = predict_disp[top_pad:, :-right_pad]
            predict_cost_softmax = predict_cost_softmax[:, top_pad:, :-right_pad]
        elif top_pad == 0 and right_pad != 0:
            predict_disp = predict_disp[:, :-right_pad]
            predict_cost_softmax = predict_cost_softmax[:, :, :-right_pad]
        elif top_pad != 0 and right_pad == 0:
            predict_disp = predict_disp[top_pad:, :]
            predict_cost_softmax = predict_cost_softmax[:, top_pad:, :]
        else:
            predict_disp = predict_disp
            predict_cost_softmax = predict_cost_softmax

        t22 = time.time()

        mask = (true_disp > 0) & (true_disp < 192)
        EPE = EPE_metric(predict_disp, true_disp, mask)
        D1 = D1_metric(predict_disp, true_disp, mask)
        Thres_1 = Thres_metric(predict_disp, true_disp, mask, 1.0)
        Thres_2 = Thres_metric(predict_disp, true_disp, mask, 2.0)
        Thres_3 = Thres_metric(predict_disp, true_disp, mask, 3.0)
        print(str(k) + ":", t22 - t11, EPE, D1, Thres_1, Thres_2, Thres_3)

        calc_avg.append([t22 - t11, EPE, D1, Thres_1, Thres_2, Thres_3])

        new_label = np.zeros_like(predict_disp)
        for y in range(predict_disp.shape[0]):
            for x in range(predict_disp.shape[1]):
                probability = predict_cost_softmax[:, y, x]
                if np.max(probability) > 0.5:
                    new_label[y, x] = predict_disp[y, x]

        obj = left_path.split("/")
        # print(obj)

        name = "/".join(obj[3:-1]) + "/"
        img_name = obj[-1]
        # print(name): flyingthings3d__frames_finalpass/frames_finalpass/TRAIN/B/0607/left/
        # print(img_name): 0008.png

        save_file = "Pseudo_stage1/" + name
        if not os.path.exists(save_file):
            os.makedirs(save_file)

        save_name = save_file + img_name[:-3] + "npy"
        print(save_name)
        np.save(save_name, new_label)


if __name__ == "__main__":
    main()





