import torch
from torch import nn
import torch.nn.functional as F
from conv_block import RRG, conv
import numpy as np


class WaveletPool(nn.Module):
    def __init__(self):
        super(WaveletPool, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class WaveletUnPool(nn.Module):
    def __init__(self):
        super(WaveletUnPool, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                            hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = torch.floor_divide(x.shape[1], 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y

class TSWT_Stage1(nn.Module):
    def __init__(self, base_channels=8):
        super(TSWT_Stage1, self).__init__()
        
        self.conv1 = conv(3, base_channels, 3)

    def forward(self, x):
        out = self.conv1(x)
        return out


class TSWT_Stage2(nn.Module):
    def __init__(self, base_channels=8, num_dab=2):
        super(TSWT_Stage2, self).__init__()

        self.conv1_s1 = RRG(base_channels, 3, base_channels//4, num_dab)
        self.conv2_s1 = RRG(base_channels, 3, base_channels//4, num_dab)
        self.conv1_s2 = RRG(base_channels*4, 3, base_channels, num_dab)

        # self.conv1_s1 = conv(base_channels, base_channels, 3)
        # self.conv2_s1 = conv(base_channels, base_channels, 3)
        # self.conv1_s2 = conv(base_channels*4, base_channels*4, 3)

        self.pool = WaveletPool()

    def forward(self, in_s1):
        out_c1s1 = self.conv1_s1(in_s1)
        out_c2s1 = self.conv2_s1(out_c1s1)
        out_p1s1 = self.pool(out_c1s1)

        out_p1s2 = self.pool(in_s1)
        out_c1s2 = self.conv1_s2(out_p1s2)

        out_s1 = out_c2s1 + in_s1
        out_s2 = out_p1s1 + out_c1s2
        return out_s1, out_s2


class TSWT_Stage3(nn.Module):
    def __init__(self, base_channels=8, num_dab=2):
        super(TSWT_Stage3, self).__init__()

        self.conv1_s1 = RRG(base_channels, 3, base_channels//4, num_dab)  
        self.conv2_s1 = RRG(base_channels, 3, base_channels//4, num_dab)  
        self.conv1_s2 = RRG(base_channels*4, 3, base_channels, num_dab)  
        self.conv2_s2 = RRG(base_channels*4, 3, base_channels, num_dab)  
        self.conv2_s3 = RRG(base_channels*16, 3, base_channels*4, num_dab)  

        # self.conv1_s1 = conv(base_channels, base_channels, 3)  
        # self.conv2_s1 = conv(base_channels, base_channels, 3)  
        # self.conv1_s2 = conv(base_channels*4, base_channels*4, 3)  
        # self.conv2_s2 = conv(base_channels*4, base_channels*4, 3)  
        # self.conv2_s3 = conv(base_channels*16, base_channels*16, 3)  

        self.pool = WaveletPool()
        self.unpool = WaveletUnPool()

    def forward(self, in_s1, in_s2):
        out_c1s1 = self.conv1_s1(in_s1)
        out_p1s1 = self.pool(in_s1)

        out_u1s2 = self.unpool(in_s2)
        out_c1s2 = self.conv1_s2(in_s2)

        out_s1 = out_c1s1 + out_u1s2
        out_s2 = out_p1s1 + out_c1s2
        out_s3 = self.pool(in_s2)

        out_c2s1 = self.conv2_s1(out_s1)
        out_p2s1 = self.pool(out_s1)

        out_u2s2 = self.unpool(out_s2)
        out_c2s2 = self.conv2_s2(out_s2)
        out_p2s2 = self.pool(out_s2)

        out_c2s3 = self.conv2_s3(out_s3)

        out_s1 = out_c2s1 + out_u2s2
        out_s2 = out_p2s1 + out_c2s2 + in_s2
        out_s3 = out_p2s2 + out_c2s3
        return out_s1, out_s2, out_s3


class TSWT_Final(nn.Module):
    def __init__(self, base_channels=8, num_dab=2):
        super(TSWT_Final, self).__init__()

        self.conv1_s1 = RRG(base_channels, 3, base_channels//4, num_dab)  

        # self.conv1_s1 = conv(base_channels, base_channels, 3)  

        self.unpool = WaveletUnPool()

    def forward(self, in_s1, in_s2, in_s3):
        out_s1 = self.conv1_s1(in_s1)
        out_s2 = self.unpool(in_s2)
        out_s3 = self.unpool(self.unpool(in_s3))
        return out_s1 + out_s2 + out_s3


class TSWT_Model(nn.Module):
    def __init__(self, base_channels=8, num_dab=2):
        super(TSWT_Model, self).__init__()
        self.stage1 = TSWT_Stage1(base_channels)
        self.stage2 = TSWT_Stage2(base_channels, num_dab)
        self.stage3 = TSWT_Stage3(base_channels, num_dab)
        self.final = TSWT_Final(base_channels, num_dab)
        self.conv_last = conv(base_channels, 3, 3)

    def forward(self, x):
        out_s1 = self.stage1(x)
        out_s1, out_s2 = self.stage2(out_s1)
        out_s1, out_s2, out_s3 = self.stage3(out_s1, out_s2)
        out = self.final(out_s1, out_s2, out_s3)
        out = self.conv_last(out) + x
        return out

