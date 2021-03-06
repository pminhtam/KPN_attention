import torch
import torch.nn as nn
import torch.nn.functional as F


def tv_loss(x, beta=0.5, reg_coeff=5):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)
    a, b, c, d = x.shape
    return reg_coeff * (torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta)) / (a * b * c * d))


class LossFunc(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        self.loss_basic = LossBasic(gradient_L1)
        self.loss_anneal = LossAnneal(alpha, beta)

    def forward(self, pred_img_i, pred_img, ground_truth, global_step):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.coeff_basic * self.loss_basic(pred_img, ground_truth), self.coeff_anneal * self.loss_anneal(global_step, pred_img_i, ground_truth)

class WaveletLoss(nn.Module):
    def __init__(self):
        super(WaveletLoss, self).__init__()
        self.pooling = WaveletPool()
        self.charbonier = CharbonnierLoss()
    def forward(self,pred,gt):
        loss = 0
        pred_LL, pred_pool = self.pooling(pred)
        gt_LL, gt_pool = self.pooling(gt)
        loss += self.charbonier(pred_pool,gt_pool)
        pred_LL_2, pred_pool_2 = self.pooling(pred)
        gt_LL_2, gt_pool_2 = self.pooling(gt)
        loss += self.charbonier(pred_pool_2,gt_pool_2)
        _, pred_pool_3 = self.pooling(pred)
        _, gt_pool_3 = self.pooling(gt)
        loss += self.charbonier(pred_pool_3,gt_pool_3)
        return loss
class WaveletPool(nn.Module):
    def __init__(self, eps=1e-3):
        super(WaveletPool, self).__init__()

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return x_LL,torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))

class LossAnneal_i(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal_i, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth[:,i,...])
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss

class LossAnneal(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal, self).__init__()
        self.global_step = 0
        self.loss_func = LossBasic(gradient_L1=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth)
        loss /= pred_i.size(1)
        return self.beta * self.alpha ** global_step * loss


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )
class BasicLoss(nn.Module):
    def __init__(self, eps=1e-3, alpha=0.998, beta=100):
        super(BasicLoss, self).__init__()
        self.charbonnier_loss = CharbonnierLoss(eps)
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, burst_pred, gt, gamma):
        b,N,c,h,w = burst_pred.size()
        burst_pred = burst_pred.view(b,c*N,h,w)
        burst_gt = torch.cat([gt[..., i::2, j::2] for i in range(2) for j in range(2)], dim=1)

        anneal_coeff = max(self.alpha ** gamma * self.beta, 1)

        burst_loss = anneal_coeff * (self.charbonnier_loss(burst_pred, burst_gt))

        single_loss = self.charbonnier_loss(pred, gt)

        loss = burst_loss + single_loss

        return loss, single_loss, burst_loss
class AlginLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(AlginLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        y = F.pad(y,[1,1,1,1])
        diff0 = torch.abs(x-y[:,:,1:-1,1:-1])
        diff1 = torch.abs(x-y[:,:,0:-2,0:-2])
        diff2 = torch.abs(x-y[:,:,0:-2,1:-1])
        diff3 = torch.abs(x-y[:,:,0:-2,2:])
        diff4 = torch.abs(x-y[:,:,1:-1,0:-2])
        diff5 = torch.abs(x-y[:,:,1:-1,2:])
        diff6 = torch.abs(x-y[:,:,2:,0:-2])
        diff7 = torch.abs(x-y[:,:,2:,1:-1])
        diff8 = torch.abs(x-y[:,:,2:,2:])
        diff_cat = torch.stack([diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8])
        diff = torch.min(diff_cat,dim=0)[0]
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
if __name__ == "__main__":
    x = torch.randn((4,3,128,128))
    print(tv_loss(x))