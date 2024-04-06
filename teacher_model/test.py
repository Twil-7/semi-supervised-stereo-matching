import time
from PIL import Image
import cv2
import os

from models import __models__, model_loss
from utils import *
import datasets.data_io
from datasets.data_io import pfm_imread

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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
    with open("model_data/test.txt") as file:
        lines = file.readlines()

    max_disp = 192
    model = __models__["gwcnet-gc"](max_disp)

    weight_path = "0098_3.59319_0.32577_0.96966_GwcNet_sceneflow.tar"
    state_dict = torch.load(weight_path)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict['state_dict'].items()})
    # model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict['model'].items()})

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = nn.DataParallel(model)
        model.cuda()

    model.eval()
    calc_avg = []
    # lines = lines[:10]
    for k in range(len(lines)):

        line = lines[k]
        obj1 = line.split('\n')    # 去除换行符
        obj2 = obj1[0].split("#")

        left_path = obj2[0]
        right_path = obj2[1]
        disp_path = obj2[2]

        img_left = Image.open(left_path).convert('RGB')
        img_right = Image.open(right_path).convert('RGB')
        true_disp, scale = pfm_imread(disp_path)

        t11 = time.time()
        processed = datasets.data_io.get_transform()
        imgL = processed(img_left)
        imgR = processed(img_right)

        # pad to width and height to 32 times
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
        Thres_1 = Thres_metric(predict_disp, true_disp, mask, 1.0)
        Thres_2 = Thres_metric(predict_disp, true_disp, mask, 2.0)
        Thres_3 = Thres_metric(predict_disp, true_disp, mask, 3.0)
        Thres_4 = Thres_metric(predict_disp, true_disp, mask, 4.0)
        Thres_5 = Thres_metric(predict_disp, true_disp, mask, 5.0)
        print(str(k) + ":", EPE, Thres_1, Thres_2, Thres_3, Thres_4, Thres_5)
        print(weight_path)
        calc_avg.append([EPE, Thres_1, Thres_2, Thres_3, Thres_4, Thres_5])

    print(np.nanmean(calc_avg, axis=0))


"""
EPE, Thres_1, Thres_2, Thres_3, Thres_4, Thres_5

1、logs/0019_6.72777_0.57249_0.94847_GwcNet_sceneflow.tar
[1.42339079 0.14443224 0.08011669 0.05930389 0.04895888 0.04250579]

2、logs/0039_5.2067_0.46684_0.95589_GwcNet_sceneflow.tar
[1.1864303  0.11986073 0.069098   0.05159301 0.04243267 0.03660319]

3、logs/0059_4.06954_0.35065_0.96695_GwcNet_sceneflow.tar
[0.91121759 0.09507543 0.05314032 0.0393413  0.03229465 0.02781428]

4、logs/0079_3.84126_0.34194_0.96819_GwcNet_sceneflow.tar
[0.88676705 0.0917918  0.05118505 0.03786318 0.03102504 0.02671769]

5、logs/0099_3.59513_0.32733_0.96947_GwcNet_sceneflow.tar
[0.86241991 0.08935883 0.04956408 0.03667063 0.03005231 0.02584561]

"""
if __name__ == "__main__":
    main()







