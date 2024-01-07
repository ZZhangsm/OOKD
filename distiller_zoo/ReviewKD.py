import torch.nn.functional as F
from torch import nn


def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all.item()

class ReviewKD(nn.Module):
    """
    Pengguang Chen, Shu Liu, Hengshuang Zhao, and Jiaya Jia.
    Distilling knowledge via knowledge review.
    In CVPR, 2021
    """
    def __init__(self):
        super(ReviewKD, self).__init__()

    def forward(self, f_s, f_t):
        loss = hcl_loss(f_s, f_t)
        return loss