import time
from PIL import Image
import cv2
import os

from models import __models__, model_loss
from utils import *
import datasets.data_io
from datasets.data_io import pfm_imread

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
    lines = lines[:200]
    for k in range(len(lines)):

        line = lines[k]
        obj1 = line.split('\n')  # 去除换行符
        obj2 = obj1[0].split("#")

        left_path = obj2[0]
        right_path = obj2[1]
        disp_path = obj2[2]

        img_left = Image.open(left_path).convert('RGB')
        img_right = Image.open(right_path).convert('RGB')
        true_disp, scale = pfm_imread(disp_path)

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

        os.makedirs("demo/", exist_ok=True)
        error_map = np.abs(predict_disp - true_disp)
        print(np.mean(error_map), np.min(error_map), np.max(error_map))
        error_map[error_map > 3.0] = 3.0
        cv2.imwrite("demo/error_" + str(k).zfill(4) + ".png", error_map / 3 * 255)

        predict_disp[predict_disp > 84] = 84
        heatmap = cv2.applyColorMap(np.uint8(255 * (predict_disp / 84)), cv2.COLORMAP_JET)
        heatmap[np.where(heatmap < 0)] = 0
        cv2.imwrite("demo/heatmap_" + str(k).zfill(4) + ".png", heatmap / np.max(heatmap) * 255)


if __name__ == "__main__":
    main()







