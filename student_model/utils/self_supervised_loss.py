import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms


class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4):
        super(MonodepthLoss, self).__init__()
        self.n = n    # 4  四个尺度
        self.image_loss = 0    # 重建rgb像素损失

    # 利用img图像和disp视差图，重建另一目的img图像
    # 右目rgb图像生成左目rgb图像，视差值为负数；左目rgb图像生成右目rgb图像，视差值为正数
    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction.
        x_shifts = disp[:, 0, :, :]
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

        # 利用新的索引和原始右目图像，相当于直接生成了预测的左目图像
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear', padding_mode='zeros')

        return output

    # 利用右目图像，生成左目图像
    def generate_image_left(self, img, disp):

        # 利用右目图像，生成左目图像，视差值得用负数，即相减
        return self.apply_disparity(img, -disp)

    # 利用左目图像，生成右目图像
    def generate_image_right(self, img, disp):

        # 利用左目图像，生成右目图像，视差值得用正数，即相加
        return self.apply_disparity(img, disp)

    def forward(self, imgL, imgR, disp_ests):

        # 对左目右目rgb图像分别进行重复四次，对应预测出来的四个视差结果
        left_pyramid = [imgL, imgL, imgL, imgL]
        right_pyramid = [imgR, imgR, imgR, imgR]

        disp_left_est = [disp.unsqueeze(1) / imgL.shape[-1] for disp in disp_ests]

        # 利用右目rgb图像+左目视差图，生成左目rgb图像；利用左目rgb图像+右目视差图，生成右目rgb图像
        # 由右目图像预测左目图像，视差图要相减；由左目图像预测右目图像，视差图要相加
        left_est = [self.generate_image_left(right_pyramid[i], disp_left_est[i]) for i in range(self.n)]

        crop_x0 = int(imgL.shape[3] * 0.15)
        # print(crop_x0)： 76

        crop_left_pyramid = [img[:, :, :, crop_x0:] for img in left_pyramid]
        crop_left_est = [img[:, :, :, crop_x0:] for img in left_est]

        """
        可视化rgb重建效果，借此看出重建图像和真实图像的差异
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        left_est0 = crop_left_est[0][0, :, :, :].cpu().detach().numpy()
        left_raw = crop_left_pyramid[0][0, :, :, :].cpu().detach().numpy()
        # print(left_est0.shape)： (3, 256, 436)
        # print(left_raw.shape)： (3, 256, 436)
        left_est0[0, :, :] = left_est0[0, :, :] * std[0] + mean[0]
        left_est0[1, :, :] = left_est0[1, :, :] * std[1] + mean[1]
        left_est0[2, :, :] = left_est0[2, :, :] * std[2] + mean[2]
        left_raw[0, :, :] = left_raw[0, :, :] * std[0] + mean[0]
        left_raw[1, :, :] = left_raw[1, :, :] * std[1] + mean[1]
        left_raw[2, :, :] = left_raw[2, :, :] * std[2] + mean[2]

        left_est0 = np.transpose(left_est0, (1, 2, 0))
        left_raw = np.transpose(left_raw, (1, 2, 0))
        # print(left_est0.shape)： (256, 436, 3)
        # print(left_raw.shape)： (256, 436, 3)
        # print(np.max(left_est0), np.min(left_est0), np.mean(left_est0))    # 1.0000001 -5.9604645e-08 0.35363415
        # print(np.max(left_raw), np.min(left_raw), np.mean(left_raw))    # 0.9490197 0.031372547 0.3562692
        
        cv2.imshow("left_est", left_est0)
        cv2.imshow("left_raw", left_raw)
        cv2.waitKey(0)
        
        """
        # 计算重建出来的左目rgb图像和真实左目rgb图像的像素差异，重建出来的右目rgb图像和真实rgb右目图像的像素差异
        left_loss = [torch.mean(torch.square(crop_left_pyramid[i] - crop_left_est[i])) for i in range(self.n)]

        image_loss = sum(left_loss)

        """
        cv2.imshow("left_est", left_est0)
        cv2.imshow("left_raw", left_raw)
        cv2.waitKey(0)
        """

        return image_loss

