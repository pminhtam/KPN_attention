from model.KPN import *
from utils.wavelet_highwave import DWT_highwave,IWT_highwave
from utils.wavelet import DWT,IWT

class retruct_basic_low(nn.Module):
    def __init__(self, in_channel,kernel_size=[5]):
        super(retruct_basic_low, self).__init__()
        self.upMode = 'bilinear'
        channel_att = True
        spatial_att = True
        in_channel = in_channel
        out_channel = np.sum(np.array(kernel_size) ** 2)
        self.color_channel = 3
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=channel_att, spatial_att=spatial_att,bn=True)
        self.conv3 = Basic(128, 256, channel_att=channel_att, spatial_att=spatial_att,bn=True)
        self.conv4 = Basic(256, 512, channel_att=channel_att, spatial_att=spatial_att,bn=True)
        # 6~8层要先上采样再卷积
        self.conv7 = Basic(256 + 512, 256, channel_att=channel_att, spatial_att=spatial_att,bn=True)
        self.conv8 = Basic(256 + 128, 128, channel_att=channel_att, spatial_att=spatial_att,bn=True)
        self.conv9 = Basic(128+64, 64, channel_att=channel_att, spatial_att=spatial_att,bn=True)
        self.outc = nn.Sequential(
            Basic(64, 64),
            nn.Conv2d(64, out_channel, 1, 1, 0)
        )
        # residual branch
        self.conv10 = Basic(128+64, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.out_res = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            Basic(64, self.color_channel, g=1),
            nn.Conv2d(self.color_channel, self.color_channel, 1, 1, 0)
        )

        self.conv11 = Basic(128+64, 128, channel_att=False, spatial_att=False)
        self.out_weight = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            Basic(64, self.color_channel, g=1),
            nn.Conv2d(self.color_channel, self.color_channel, 1, 1, 0),
            # nn.Softmax(dim=1)  #softmax 效果较差
            nn.Sigmoid()
        )
        self.kernel_pred = KernelConv(kernel_size, False, core_bias=False)

    def forward(self, data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))

        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv4, scale_factor=2, mode=self.upMode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        conv9 = self.conv9(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        core = self.outc(conv9)
        # residual branch
        conv10 = self.conv10(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        residual = self.out_res(conv10)

        conv11 = self.conv11(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        weight = self.out_weight(conv11)

        pred_i, _ = self.kernel_pred(data, core, 1.0)
        # only for gray images now, supporting for RGB could be programed later

        pred_i, _ = self.kernel_pred(data.unsqueeze(0), core, 1.0)
        pred = pred_i[0]
        weight = weight.view(pred.size())
        residual = residual.view(pred.size())
        pred = weight*pred + (1-weight)*residual

        return pred
class retruct_basic_high(nn.Module):
    def __init__(self, in_channel,kernel_size=[5]):
        super(retruct_basic_high, self).__init__()
        self.upMode = 'bilinear'
        channel_att = True
        spatial_att = True
        in_channel = in_channel
        out_channel = np.sum(np.array(kernel_size) ** 2)

        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False,bn=True)
        self.conv2 = Basic(64, 128, channel_att=channel_att, spatial_att=spatial_att,bn=True)
        self.conv3 = Basic(128, 256, channel_att=channel_att, spatial_att=spatial_att,bn=True)
        self.conv4 = Basic(256, 256, channel_att=channel_att, spatial_att=spatial_att,bn=True)

        self.outc = nn.Sequential(
            Basic(256, 64, channel_att=channel_att, spatial_att=spatial_att,bn=True),
            Basic(64, 64),
            nn.Conv2d(64, out_channel, 1, 1, 0)
        )
        self.kernel_pred = KernelConv(kernel_size, False, core_bias=False)

    def forward(self, data):
        conv1 = self.conv1(data)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        core = self.outc(conv4)
        pred_i, _ = self.kernel_pred(data.unsqueeze(0), core, 1.0)
        return pred_i[0]

class Att_KPN_Wavelet_highwave(nn.Module):
    def __init__(self, color=True, burst_length=8, channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(Att_KPN_Wavelet_highwave, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        # in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length + 1)
        in_channel = (3 if color else 1)

        out_channel = (3 if color else 1)
        self.DWT = DWT_highwave()
        self.IWT = IWT_highwave()
        # 各个卷积层定义
        # 2~5 Down image
        n_feat = 64
        # self.conv1 = Basic(in_channel, n_feat, channel_att=channel_att, spatial_att=spatial_att)
        print(in_channel)
        self.low_1 = retruct_basic_low(in_channel)
        self.high_hh1 = retruct_basic_high(in_channel)
        self.high_hl1 = retruct_basic_high(in_channel)
        self.high_lh1 = retruct_basic_high(in_channel)
        self.low_2 = retruct_basic_low(in_channel)
        self.high_hh2 = retruct_basic_high(in_channel)
        self.high_hl2 = retruct_basic_high(in_channel)
        self.high_lh2 = retruct_basic_high(in_channel)
        # self.low_3 = retruct_basic_low(in_channel)
        # self.high_hh3 = retruct_basic_high(in_channel)
        # self.high_lh3 = retruct_basic_high(in_channel)
        # self.high_hl3 = retruct_basic_high(in_channel)

        self.outc = nn.Sequential(
            Basic(in_channel, 64, channel_att=channel_att, spatial_att=spatial_att),
            Basic(64, 64),
            nn.Conv2d(64, out_channel, 1, 1, 0)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if type(m.bias) != type(None):
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        x_LL1, x_HL1, x_LH1, x_HH1 = self.DWT(data)
        x_HL1 = self.high_hl1(x_HL1)
        x_HH1 = self.high_hh1(x_HH1)
        x_LH1 = self.high_lh1(x_LH1)

        x_LL2, x_HL2, x_LH2, x_HH2 = self.DWT(x_LL1)
        x_HL2 = self.high_hl1(x_HL2)
        x_HH2 = self.high_hh1(x_HH2)
        x_LH2 = self.high_lh1(x_LH2)

        # x_LL3, x_HL3, x_LH3, x_HH3 = self.DWT(x_LL2)
        # x_HL3 = self.high_hl1(x_HL3)
        # x_HH3 = self.high_hh1(x_HH3)
        # x_LH3 = self.high_lh1(x_LH3)
        # x_LL3 = self.low_3(x_LL3)
        # x_LL2 = self.IWT(torch.cat([x_LL3, x_HL3, x_LH3, x_HH3],dim=1))
        x_LL2 = self.low_2(x_LL2)
        x_LL1 = self.IWT(torch.cat([x_LL2, x_HL2, x_LH2, x_HH2],dim=1))
        x_LL1 = self.low_1(x_LL1)
        x = self.IWT(torch.cat([x_LL1, x_HL1, x_LH1, x_HH1],dim=1))
        out = self.outc(x)
        return out

if __name__ == "__main__":
    model = Att_KPN_Wavelet_highwave(
        color=True,
        burst_length=1,
        channel_att=True,
        spatial_att=True,
        upMode="bilinear",
        core_bias=False,
    )
    print(model)