from model.KPN import *
from utils.wavelet import DWT,IWT


class Att_NonKPN_Wavelet(nn.Module):
    def __init__(self, color=True, burst_length=8, blind_est=False,kernel_size=[3],
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False,bn=False):
        super(Att_NonKPN_Wavelet, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length + 1)

        out_channel = (3 if color else 1)
        self.DWT = DWT()
        self.IWT = IWT()
        # 各个卷积层定义
        # 2~5 Down image
        n_feat = 64
        self.conv1 = Basic(in_channel, n_feat, channel_att=channel_att, spatial_att=spatial_att)
        self.conv2 = Basic(n_feat*4, n_feat*2, channel_att=channel_att, spatial_att=spatial_att)
        self.conv3 = Basic(n_feat*2*4, n_feat*2*2, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv4 = Basic(n_feat*2*2*4, n_feat*2*2*2, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv5 = Basic(n_feat*2*2*2*4, n_feat*2*2*2*4, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        # 6~8 Up image
        self.conv6 = Basic(n_feat*2*2*4, n_feat*2*2*4, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv7 = Basic(n_feat*2*4, n_feat*2*4, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv8 = Basic(n_feat*4, n_feat*4, channel_att=channel_att, spatial_att=spatial_att)
        self.conv9 = Basic(n_feat*2, n_feat, channel_att=channel_att, spatial_att=spatial_att)

        self.kernel_size = kernel_size[0]
        self.outc = nn.Sequential(
            Basic(n_feat, n_feat),
            nn.Conv2d(n_feat, out_channel, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=self.core_bias)
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

        return core