from model.KPN import *
from utils.wavelet import IWT
from utils.wavelet_highwave import DWT_highwave


class Att_NonKPN_Wavelet_highwave(nn.Module):
    def __init__(self, color=True, burst_length=8, blind_est=False,kernel_size=[3],
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False,bn=False):
        super(Att_NonKPN_Wavelet_highwave, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        # in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length + 1)
        in_channel = (3 if color else 1)

        out_channel = (3 if color else 1)
        self.DWT = DWT_highwave()
        self.IWT = IWT()
        # 各个卷积层定义
        # 2~5 Down image
        n_feat = in_channel
        # self.conv1 = Basic(in_channel, n_feat, channel_att=channel_att, spatial_att=spatial_att)
        print(in_channel)
        self.conv2 = Basic(in_channel*3, in_channel*3, g=in_channel ,channel_att=channel_att, spatial_att=spatial_att)
        self.conv1_low = Basic(in_channel, in_channel, g=in_channel ,channel_att=channel_att, spatial_att=spatial_att)
        self.conv3 = Basic(in_channel*3*3, in_channel*3*3, g=in_channel, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv2_low = Basic(in_channel*3, in_channel*3, g=in_channel, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv4 = Basic(in_channel*3*3*3,in_channel*3*3*3, g=in_channel*3, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv3_low = Basic(in_channel*3*3,in_channel*3*3, g=in_channel*3, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv5 = Basic(in_channel*3*3*3*3, in_channel*3*3*3*3, g=in_channel*3*3, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        # 6~8 Up image
        self.conv6 = Basic(in_channel*3*3*3, in_channel*3*3*3, g=in_channel*3, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv7 = Basic(in_channel*3*3, in_channel*3*3, g=in_channel, channel_att=channel_att, spatial_att=spatial_att,bn=bn)
        self.conv8 = Basic(in_channel*3, in_channel*3, g=in_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.conv9 = Basic(in_channel, in_channel, g=in_channel, channel_att=channel_att, spatial_att=spatial_att)

        self.kernel_size = kernel_size[0]
        # self.outc = nn.Sequential(
        #     Basic(n_feat, n_feat),
        #     nn.Conv2d(n_feat, out_channel, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=self.core_bias)
        # )


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
    def forward(self, data_with_est):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        # conv1 = self.conv1(data_with_est) # 3*1024*1024
        low_conv1, high_conv1 = self.DWT(data_with_est) # 3*512*512, 3*3*512*512
        conv2 = self.conv2(high_conv1)  # 3*3*512*512
        low_conv1 = self.conv1_low(low_conv1)
        low_conv2, high_conv2 = self.DWT(conv2) # 3*3*256*256 , 3*3*3*256*256
        conv3 = self.conv3(high_conv2) # 3*3*3*256*256
        low_conv2 = self.conv2_low(low_conv2)
        low_conv3, high_conv3 = self.DWT(conv3) # 3*3*3*128*128 , 3*3*3*3*128*128
        conv4 = self.conv4(high_conv3) # 3*3*3*3*128*128
        low_conv3 = self.conv3_low(low_conv3)
        low_conv4, high_conv4 = self.DWT(conv4) # 3*3*3*3*64*64 , 3*3*3*3*3*64*64
        conv5 = self.conv5(high_conv4) # 3*3*3*3*3*64*64
        # 开始上采样  同时要进行skip connection
        # conv5 = 3*3*3*3*3*64*64
        conv6 = self.conv6(self.IWT(torch.cat([low_conv4,conv5],dim=1)))
        # conv6 = 3*3*3*3*128*128
        conv7 = self.conv7(self.IWT(torch.cat([low_conv3,conv6],dim=1)))
        # conv7 = 3*3*3*256*256
        conv8 = self.conv8(self.IWT(torch.cat([low_conv2,conv7],dim=1)))
        # conv8 = 3*3*512*512
        out = self.IWT(torch.cat([low_conv1,conv8],dim=1))
        # out = 3*1024*1024
        # return channel K*K*N
        # core = self.outc(conv9)

        return out

if __name__ == "__main__":
    model = Att_NonKPN_Wavelet_highwave(
        color=True,
        burst_length=1,
        blind_est=True,
        kernel_size=[3],
        channel_att=True,
        spatial_att=True,
        upMode="bilinear",
        core_bias=False,
        bn=True
    )
    print(model)