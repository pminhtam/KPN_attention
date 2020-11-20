import torch
from utils.KPN import KPN
from utils.Att_KPN import Att_KPN
from utils.Att_Weight_KPN import Att_Weight_KPN
import torch.nn as nn
from guided_filter_pytorch.guided_filter import ConvGuidedFilter
from utils.noise_estimation import Network as NoiseEstimate

class KPN_noise(nn.Module):
    def __init__(self,color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN_noise, self).__init__()
        self.KPN = KPN(
            color=color,
            burst_length=burst_length,
            blind_est=blind_est,
            kernel_size=kernel_size,
            sep_conv=sep_conv,
            channel_att=channel_att,
            spatial_att=spatial_att,
            upMode=upMode,
            core_bias=core_bias
        )
        self.noise_estimate = NoiseEstimate(color=color)

    def forward(self,data):
        noise = self.noise_estimate(data[:,0,:,:])

        b, N, c, h, w = data.size()
        feedData = data.view(b, -1, h, w)
        feedData = torch.cat([feedData,noise],dim=1)
        pred_i, pred = self.KPN(feedData, data)

        return pred_i,pred

class Att_KPN_noise(nn.Module):
    def __init__(self,color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(Att_KPN_noise, self).__init__()
        self.Att_KPN = Att_KPN(
            color=color,
            burst_length=burst_length,
            blind_est=blind_est,
            kernel_size=kernel_size,
            sep_conv=sep_conv,
            channel_att=channel_att,
            spatial_att=spatial_att,
            upMode=upMode,
            core_bias=core_bias
        )
        self.noise_estimate = NoiseEstimate(color=color)

    def forward(self, data):
        noise = self.noise_estimate(data[:,0,:,:])
        b, N, c, h, w = data.size()
        feedData = data.view(b, -1, h, w)
        feedData = torch.cat([feedData,noise],dim=1)
        pred_i, pred = self.Att_KPN(feedData, data)

        return pred_i,pred

class Att_Weight_KPN_noise(nn.Module):
    def __init__(self,color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(Att_Weight_KPN_noise, self).__init__()
        self.Att_Weight_KPN = Att_Weight_KPN(
            color=color,
            burst_length=burst_length,
            blind_est=blind_est,
            kernel_size=kernel_size,
            sep_conv=sep_conv,
            channel_att=channel_att,
            spatial_att=spatial_att,
            upMode=upMode,
            core_bias=core_bias
        )
        self.noise_estimate = NoiseEstimate(color=color)

    def forward(self, data):
        noise = self.noise_estimate(data[:,0,:,:])
        b, N, c, h, w = data.size()
        feedData = data.view(b, -1, h, w)
        feedData = torch.cat([feedData,noise],dim=1)
        pred_i, pred = self.Att_Weight_KPN(feedData, data)

        return pred_i,pred