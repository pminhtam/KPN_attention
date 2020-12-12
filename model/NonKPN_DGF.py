from model.NonKPN import Att_NonKPN_Wavelet
import torch.nn as nn
from guided_filter_pytorch.guided_filter import ConvGuidedFilter


class Att_NonKPN_Wavelet_DGF(nn.Module):
    def __init__(self,color=True, burst_length=8, blind_est=False, kernel_size=[3], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False,bn=False):
        super(Att_NonKPN_Wavelet_DGF, self).__init__()
        self.Att_NonKPN_Wavelet = Att_NonKPN_Wavelet(
            color=color,
            burst_length=burst_length,
            blind_est=blind_est,
            kernel_size=kernel_size,
            channel_att=channel_att,
            spatial_att=spatial_att,
            upMode=upMode,
            core_bias=core_bias,
            bn=bn
        )
        self.gf = ConvGuidedFilter(radius=1)

    def forward(self,data_with_est, data,x_hr):
        pred = self.Att_NonKPN_Wavelet(data_with_est, data)
        b, N, c, h, w = data.size()
        data_feed = data[:,0,:,:,:].view(-1,c, h, w)
        pred_feed = pred.view(-1,c, h, w)

        b_hr, c_hr, h_hr, w_hr = x_hr.size()
        x_hr_feed = x_hr.view(-1,c, h_hr, w_hr)
        out_hr = self.gf(data_feed, pred_feed, x_hr_feed)

        out_hr = out_hr.view(b,c, h_hr,w_hr)
        return out_hr