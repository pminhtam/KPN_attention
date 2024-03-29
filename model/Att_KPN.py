from model.KPN import *
from utils.wavelet import DWT,IWT

class Att_KPN(nn.Module):
    def __init__(self, color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False,in_channel = 3):
        super(Att_KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        # in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length + 1)
        # in_channel = in_channel*burst_length
        in_channel = in_channel
        out_channel = (
            2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length

        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512 + 512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256 + 512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256 + 128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.conv9 = Basic(out_channel+64, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Sequential(
            Basic(out_channel, out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        )

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        conv9 = self.conv9(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        core = self.outc(conv9)

        return self.kernel_pred(data, core, white_level)
        # return core,conv9
class Att_KPN_Wavelet(nn.Module):
    def __init__(self, color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False,in_channel = 3):
        super(Att_KPN_Wavelet, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length*burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        # in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length + 1)
        in_channel = in_channel

        out_channel = (
            2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        self.DWT = DWT()
        self.IWT = IWT()
        # 各个卷积层定义
        # 2~5 Down image
        n_feat = 64
        self.conv1 = Basic(in_channel, n_feat, channel_att=channel_att, spatial_att=spatial_att)
        self.conv2 = Basic(n_feat*4, n_feat*2, channel_att=channel_att, spatial_att=spatial_att)
        self.conv3 = Basic(n_feat*2*4, n_feat*2*2, channel_att=channel_att, spatial_att=spatial_att)
        self.conv4 = Basic(n_feat*2*2*4, n_feat*2*2*2, channel_att=channel_att, spatial_att=spatial_att)
        self.conv5 = Basic(n_feat*2*2*2*4, n_feat*2*2*2*4, channel_att=channel_att, spatial_att=spatial_att)
        # 6~8 Up image
        self.conv6 = Basic(n_feat*2*2*4, n_feat*2*2*4, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(n_feat*2*4, n_feat*2*4, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(n_feat*4, n_feat*4, channel_att=channel_att, spatial_att=spatial_att)
        self.conv9 = Basic(n_feat*2, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Sequential(
            Basic(out_channel, out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        )

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(self.DWT(conv1))
        conv3 = self.conv3(self.DWT(conv2))
        conv4 = self.conv4(self.DWT(conv3))
        conv5 = self.conv5(self.DWT(conv4))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, self.IWT(conv5)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, self.IWT(conv6)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, self.IWT(conv7)], dim=1))
        conv9 = self.conv9(torch.cat([conv1, self.IWT(conv8)], dim=1))
        # return channel K*K*N
        core = self.outc(conv9)

        return self.kernel_pred(data, core, white_level)
if __name__ == '__main__':
    kpn = Att_KPN_Wavelet(
            color=True,
            burst_length=4,
            blind_est=True,
            kernel_size=[5],
            sep_conv=False,
            channel_att=True,
            spatial_att=True,
            upMode="bilinear",
            core_bias=False
        )
    # print(kpn)
    print(summary(kpn, [(12, 224, 224),(4,3, 224, 224)], batch_size=1))
