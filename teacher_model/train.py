from __future__ import print_function, division
import argparse
import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import time
from torch.utils.data import DataLoader
import gc    

from datasets import __datasets__
from models import __models__, model_loss
from utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
# os.environ['CUDA_VISIBLE_DEVICES'] = "6"

# 设置torch.backends.cudnn.benchmark=True，会让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，为整个网络的每个
# 卷积层搜索最适合它的卷积实现算法，进而实现网络的加速.
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-gc', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default="sceneflow", help='dataset name', choices=__datasets__.keys())
parser.add_argument('--trainlist', default="model_data/labeled.txt", help='training list')
parser.add_argument('--testlist', default="model_data/test.txt", help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="50:10", help='the epochs to decay lr: the downscale rate')
parser.add_argument('--logdir', default="logs/", help='the directory to save logs and checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.trainlist, True)
test_dataset = StereoDataset(args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

start_epoch = 0


def main():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        avg_train_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            # print(set(sample))： {'left', 'disparity', 'right'}
            # print(sample['left'].shape)： torch.Size([1, 3, 256, 512])
            # print(sample['right'].shape)： torch.Size([1, 3, 256, 512])
            # print(sample['disparity'].shape)： torch.Size([1, 256, 512])

            start_time = time.time()
            loss, scalar_outputs, image_outputs = train_sample(sample)

            avg_train_scalars.update(scalar_outputs)

            del scalar_outputs, image_outputs
            if batch_idx % 100 == 0:
                print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'
                      .format(epoch_idx, args.epochs,  batch_idx, len(TrainImgLoader), loss, time.time() - start_time))

        avg_train_scalars = avg_train_scalars.mean()
        print("avg_train_scalars", avg_train_scalars)

        gc.collect()    # 强制对所有代进行垃圾回收

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):

            start_time = time.time()
            loss, scalar_outputs, image_outputs = test_sample(sample)

            avg_test_scalars.update(scalar_outputs)

            del scalar_outputs, image_outputs
            if batch_idx % 100 == 0:
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.
                      format(epoch_idx, args.epochs, batch_idx, len(TestImgLoader), loss, time.time() - start_time))
                # Epoch 0/300, Iter 0/20, test loss = 17.020, time = 1.741915

        avg_test_scalars = avg_test_scalars.mean()
        print("avg_test_scalars", avg_test_scalars)

        gc.collect()

        train_loss = avg_train_scalars['loss']
        val_loss = avg_test_scalars['loss']
        d1_acc = 1 - avg_test_scalars['D1'][0]

        train_loss = np.round(train_loss, 5)
        val_loss = np.round(val_loss, 5)
        d1_acc = np.round(d1_acc, 5)

        save_name = args.logdir + str(epoch_idx).zfill(4) + "_" + str(train_loss) + "_" + str(val_loss) + "_" \
                    + str(d1_acc) + "_GwcNet_sceneflow.tar"

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'average_train_loss': train_loss,
            'average_val_loss': val_loss,
            'epoch': epoch_idx + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': d1_acc}, save_name)


# train one sample
def train_sample(sample):
    # print(sample['left'].shape)： torch.Size([1, 3, 256, 512])
    # print(sample['right'].shape)： torch.Size([1, 3, 256, 512])
    # print(sample['disparity'].shape)： torch.Size([1, 256, 512])
    # print(compute_metrics)： True

    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    # print(imgL.shape)： torch.Size([1, 3, 256, 512])
    # print(imgR.shape)： torch.Size([1, 3, 256, 512])
    # print(disp_gt.shape)： torch.Size([1, 256, 512])
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    # print(len(disp_ests))： 4
    # print(disp_ests[0].shape, disp_ests[1].shape, disp_ests[2].shape, disp_ests[3].shape)
    # torch.Size([1, 256, 512]) torch.Size([1, 256, 512]) torch.Size([1, 256, 512]) torch.Size([1, 256, 512])

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)
    # print(loss)： tensor(157.5112, device='cuda:0', grad_fn=<AddBackward0>)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    # print(imgL.shape)： torch.Size([1, 3, 384, 1248])
    # print(imgR.shape)： torch.Size([1, 3, 384, 1248])
    # print(disp_gt.shape)： torch.Size([1, 384, 1248])

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests, _ = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)
    # print(loss)： tensor(17.0196, device='cuda:0')

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    main()
