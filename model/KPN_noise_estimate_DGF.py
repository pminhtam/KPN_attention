from model.KPN_noise_estimate import KPN_noise,Att_KPN_noise,Att_Weight_KPN_noise
import torch.nn as nn
from guided_filter_pytorch.guided_filter import ConvGuidedFilter


class KPN_noise_DGF(nn.Module):
    def __init__(self,color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(KPN_noise_DGF, self).__init__()
        self.KPN_noise = KPN_noise(
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
        self.gf = ConvGuidedFilter(radius=1)

    def forward(self,data,x_hr):
        pred_i, pred,noise = self.KPN_noise(data)
        b, N, c, h, w = data.size()
        data_feed = data[:,0,:,:,:].view(-1,c, h, w)
        pred_feed = pred.view(-1,c, h, w)
        b_hr, c_hr, h_hr, w_hr = x_hr.size()
        x_hr_feed = x_hr.view(-1,c, h_hr, w_hr)
        out_hr = self.gf(data_feed, pred_feed, x_hr_feed)
        out_hr = out_hr.view(b,c, h_hr,w_hr)
        return pred_i,out_hr,noise
class Att_KPN_noise_DGF(nn.Module):
    def __init__(self,color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(Att_KPN_noise_DGF, self).__init__()
        self.Att_KPN_noise = Att_KPN_noise(
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
        self.gf = ConvGuidedFilter(radius=1)

    def forward(self, data,x_hr):
        pred_i, pred,noise = self.Att_KPN_noise(data)
        b, N, c, h, w = data.size()
        data_feed = data[:, 0, :, :, :].view(-1, c, h, w)
        pred_feed = pred.view(-1, c, h, w)
        b_hr, c_hr, h_hr, w_hr = x_hr.size()
        x_hr_feed = x_hr.view(-1, c, h_hr, w_hr)
        out_hr = self.gf(data_feed, pred_feed, x_hr_feed)
        out_hr = out_hr.view(b, c, h_hr, w_hr)
        return pred_i,out_hr,noise

class Att_Weight_KPN_noise_DGF(nn.Module):
    def __init__(self,color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(Att_Weight_KPN_noise_DGF, self).__init__()
        self.Att_Weight_KPN_noise = Att_Weight_KPN_noise(
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
        self.gf = ConvGuidedFilter(radius=1)

    def forward(self, data,x_hr):
        pred_i, pred,noise = self.Att_Weight_KPN_noise(data)
        b, N, c, h, w = data.size()
        data_feed = data[:, 0, :, :, :].view(-1, c, h, w)
        pred_feed = pred.view(-1, c, h, w)
        b_hr, c_hr, h_hr, w_hr = x_hr.size()
        x_hr_feed = x_hr.view(-1, c, h_hr, w_hr)
        out_hr = self.gf(data_feed, pred_feed, x_hr_feed)
        out_hr = out_hr.view(b, c, h_hr, w_hr)
        return pred_i,out_hr,noise