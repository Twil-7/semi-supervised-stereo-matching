from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    # print(x.shape, maxdisp)： torch.Size([1, 192, 256, 512]) 192

    assert len(x.shape) == 4

    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    # print(disp_values.shape)： torch.Size([192])
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    # print(disp_values.shape)： torch.Size([1, 192, 1, 1])

    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    # print(refimg_fea.shape): torch.Size([1, 12, 64, 128])
    # print(targetimg_fea.shape): torch.Size([1, 12, 64, 128])
    # print(maxdisp): 48

    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    # print(volume.shape)： torch.Size([1, 24, 48, 64, 128])

    # 普通concat操作，计算cost volume
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()

    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    # print(fea1.shape)： torch.Size([1, 320, 64, 128])
    # print(fea2.shape)： torch.Size([1, 320, 64, 128])
    # print(num_groups)： 40

    B, C, H, W = fea1.shape    # B=1、C=320、H=64、W=128
    assert C % num_groups == 0    # 必须要求feature map的维度是分组的整数倍
    channels_per_group = C // num_groups    # 320 // 40 = 8

    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    # print(cost.shape)： torch.Size([1, 40, 64, 128])

    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    # print(refimg_fea.shape)： torch.Size([1, 320, 64, 128])
    # print(targetimg_fea.shape)： torch.Size([1, 320, 64, 128])
    # print(maxdisp)： 48
    # print(num_groups)： 40

    B, C, H, W = refimg_fea.shape    # B=1、C=320、H=64、W=128

    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    # print(volume.shape): torch.Size([1, 40, 48, 64, 128])

    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:],
                                                           targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        # print(inplanes, planes, stride, downsample, pad, dilation)： 32 32 1 None 1 1

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
