import torch
from torch import nn
from conv_block import conv
from tswt import TSWT_Stage1, TSWT_Stage2, TSWT_Stage3, TSWT_Final



class TSWT3D_Model(nn.Module):
    def __init__(self, base_channels=32, num_dab=2, cache_percent=0.125):
        super(TSWT3D_Model, self).__init__()
        self.addition = int(base_channels * cache_percent)

        self.stage1 = TSWT_Stage1(base_channels - self.addition)
        self.stage2 = TSWT_Stage2(base_channels, num_dab)
        self.stage3 = TSWT_Stage3(base_channels + self.addition, num_dab)
        self.final = TSWT_Final(base_channels + self.addition*2, num_dab)
        self.last_conv = conv(base_channels + self.addition*2, 3, 3)


    def transfer_cache(self, feature_map, cache, num_cache_channels):
        next_cache = feature_map[:, -num_cache_channels:, :, :]

        feature_map = torch.cat([feature_map, cache], dim=1)
        return feature_map, next_cache

    def inference(self, x, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32, cache_s33):
        out_s1 = self.stage1(x)
        out_s1, next_cache_s1 = self.transfer_cache(out_s1, cache_s1, 1*self.addition)

        out_s1, out_s2 = self.stage2(out_s1)
        out_s1, next_cache_s21 = self.transfer_cache(out_s1, cache_s21, 1*self.addition)
        out_s2, next_cache_s22 = self.transfer_cache(out_s2, cache_s22, 4*self.addition)

        out_s1, out_s2, out_s3 = self.stage3(out_s1, out_s2)
        out_s1, next_cache_s31 = self.transfer_cache(out_s1, cache_s31, 1*self.addition)
        out_s2, next_cache_s32 = self.transfer_cache(out_s2, cache_s32, 4*self.addition)
        out_s3, next_cache_s33 = self.transfer_cache(out_s3, cache_s33, 16*self.addition)

        out = self.final(out_s1, out_s2, out_s3)
        out = self.last_conv(out) + x
        return out, next_cache_s1, next_cache_s21, next_cache_s22, next_cache_s31, next_cache_s32, \
            next_cache_s33


class DenoiseNet(nn.Module):
    def __init__(self, base_channels=32, num_dab=2, cache_percent=0.125):
        super(DenoiseNet, self).__init__()
        self.tswt           = TSWT3D_Model(base_channels, num_dab, cache_percent)
        self.cache_percent  = cache_percent
        self.base_channels  = base_channels

    def forward(self, noisy0, noisy1, noisy2, noisy3, noisy):
        cache_percent   = self.cache_percent
        base_channels   = self.base_channels

        batch_size = noisy.shape[0]
        H, W = noisy.shape[-2:]
        
        img_size1 = (H // 2, W // 2)
        img_size2 = (H // 4, W // 4)
        img_size3 = (H // 8, W // 8)

        channel1 = int(base_channels * cache_percent)
        channel2 = channel1 * 4
        channel3 = channel1 * 16

        cache_s1 = torch.zeros( (batch_size, channel1) + img_size1 )
        cache_s21 = torch.zeros( (batch_size, channel1) + img_size1 )
        cache_s31 = torch.zeros( (batch_size, channel1) + img_size1 )
        cache_s22 = torch.zeros( (batch_size, channel2) + img_size2 )
        cache_s32 = torch.zeros( (batch_size, channel2) + img_size2 )
        cache_s33 = torch.zeros( (batch_size, channel3) + img_size3)
        
        list_denoised_lr_images = []

        noisy_batch = noisy0
        denoised_lr_image, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32, cache_s33 = \
            self.tswt.inference(noisy_batch, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32,
                                cache_s33)
        list_denoised_lr_images.append(denoised_lr_image)
        
        noisy_batch = noisy1
        denoised_lr_image, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32, cache_s33 = \
            self.tswt.inference(noisy_batch, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32,
                                cache_s33)
        list_denoised_lr_images.append(denoised_lr_image)

        noisy_batch = noisy2
        denoised_lr_image, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32, cache_s33 = \
            self.tswt.inference(noisy_batch, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32,
                                cache_s33)
        list_denoised_lr_images.append(denoised_lr_image)

        noisy_batch = noisy3
        denoised_lr_image, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32, cache_s33 = \
            self.tswt.inference(noisy_batch, cache_s1, cache_s21, cache_s22, cache_s31, cache_s32,
                                cache_s33)
        list_denoised_lr_images.append(denoised_lr_image)

        return torch.mean(torch.stack(list_denoised_lr_images, dim=1), dim=1)
