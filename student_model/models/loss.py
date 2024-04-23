import torch.nn.functional as F


def model_loss(disp_ests, disp_gt, mask):
    # print(len(disp_ests))： 4
    # print(disp_ests[0].shape, disp_ests[1].shape, disp_ests[2].shape, disp_ests[3].shape)：
    # torch.Size([1, 256, 512]) torch.Size([1, 256, 512]) torch.Size([1, 256, 512]) torch.Size([1, 256, 512])
    # print(disp_gt.shape)： torch.Size([1, 256, 512])
    # print(mask.shape)： torch.Size([1, 256, 512])

    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    # print(all_losses)
    # [tensor(34.5496, device='cuda:0', grad_fn=<MulBackward0>),
    #  tensor(35.1372, device='cuda:0', grad_fn=<MulBackward0>),
    #  tensor(37.2556, device='cuda:0', grad_fn=<MulBackward0>),
    #  tensor(50.5687, device='cuda:0', grad_fn=<MulBackward0>)]

    return sum(all_losses)
