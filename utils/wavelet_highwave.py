import math

import torch
import torch.nn as nn
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2] /2
    x2 = x02[:, :, :, 0::2] /2
    x3 = x01[:, :, :, 1::2] /2
    x4 = x02[:, :, :, 1::2] /2
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH



def iwt_init(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :]
    x2 = x[:, out_channel:out_channel * 2, :, :]
    x3 = x[:, out_channel * 2:out_channel * 3, :, :]
    x4 = x[:, out_channel * 3:out_channel * 4, :, :]


    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT_highwave(nn.Module):
    def __init__(self):
        super(DWT_highwave, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT_highwave(nn.Module):
    def __init__(self):
        super(IWT_highwave, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
