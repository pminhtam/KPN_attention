import argparse
import os
import torch
# from utils.training_util import load_checkpoint
# from utils.data_provider_DGF import pixel_unshuffle
# from model.KPN_noise_estimate_DGF import KPN_noise_DGF,Att_KPN_noise_DGF,Att_Weight_KPN_noise_DGF
# from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image
import time
from utils.training_util import calculate_psnr, calculate_ssim
import math
def dwt(x):

    x01 = x[ :, 0::2, :] / 2
    x02 = x[ :, 1::2, :] / 2
    x1 = x01[:, :, 0::2] /2
    x2 = x02[ :, :, 0::2] /2
    x3 = x01[:, :, 1::2] /2
    x4 = x02[:, :, 1::2]/ 2
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH

def iwt(x_LL, x_HL, x_LH, x_HH):

    x1 = x_LL
    x2 = x_HL
    x3 = x_LH
    x4 = x_HH
    in_channel, in_height, in_width = x_LL.size()
    out_channel, out_height, out_width = in_channel, 2 * in_height, 2 * in_width

    h = torch.zeros([out_channel, out_height, out_width]).float()

    h[:, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[ :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

def test_wave(args):
    noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    clean_path = [ i.replace("noisy","clean") for i in noisy_path]
    image_file = noisy_path[2]
    image_noise = transforms.ToTensor()(Image.open(image_file).convert('RGB'))
    gt = transforms.ToTensor()(Image.open(clean_path[2]).convert('RGB'))
    x_LL, x_HL, x_LH, x_HH = dwt(image_noise)
    x_LL2, x_HL2, x_LH2, x_HH2 = dwt(x_HL)
    x_LL3, x_HL3, x_LH3, x_HH3 = dwt(x_LL2)
    x_LL4, x_HL4, x_LH4, x_HH4 = dwt(x_LL3)
    # x_LL5, x_HL5, x_LH5, x_HH5 = dwt(x_LL4)


    x_LL_gt, x_HL_gt, x_LH_gt, x_HH_gt = dwt(image_noise)
    # x_LL_gt2, x_HL_gt2, x_LH_gt2, x_HH_gt2 = dwt(x_HL_gt)
    # x_LL_gt3, x_HL_gt3, x_LH_gt3, x_HH_gt3 = dwt(x_LL_gt2)
    # x_LL_gt4, x_HL_gt4, x_LH_gt4, x_HH_gt4 = dwt(x_LL_gt3)
    # x_LL, x_HL, x_LH, x_HH = dwt(x_LL)
    # x_LL, x_HL, x_LH, x_HH = dwt(x_LL)
    # x_LL, x_HL, x_LH, x_HH = dwt(x_LL)

    trans = transforms.ToPILImage()
    # denoise_LL3 = iwt(x_LL_gt4,x_HL_gt4, x_LH_gt4, x_HH_gt4)
    # denoise_LL2 = iwt(denoise_LL3,x_HL_gt3, x_LH_gt3, x_HH_gt3)
    # denoise_LL1 = iwt(denoise_LL2,x_HL_gt2, x_LH_gt2, x_HH_gt2)
    # denoise_LL = iwt(x_LL,x_HL_gt, x_LH_gt, x_HH_gt)
    # psnr_t = calculate_psnr(denoise_LL, gt)
    # ssim_t = calculate_ssim(denoise_LL, gt)
    # print("   UP   :  PSNR : ", str(psnr_t), " :  SSIM : ", str(ssim_t))
    plt.figure(figsize=(30, 9))
    plt.subplot(2, 3, 1)
    plt.imshow(np.array(trans(x_LL_gt)))
    plt.title("x_LH " + args.model_type, fontsize=12)
    plt.subplot(2, 3, 2)
    plt.imshow(np.array(trans(x_HL_gt)))
    plt.title("x_HH_gt " + args.model_type, fontsize=12)
    plt.subplot(2, 3, 3)
    plt.imshow(np.array(trans(x_LH_gt)))
    plt.title("x_HH2 " + args.model_type, fontsize=12)
    plt.subplot(2, 3, 4)
    plt.imshow(np.array(trans(x_HH_gt)))
    plt.title("x_HH_gt2 " + args.model_type, fontsize=12)
    plt.show()

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/FullTest/noisy', help='path to noise image file')
    parser.add_argument('--gt','-g', default='/home/dell/Downloads/FullTest/clean', help='path to noise image file')
    parser.add_argument('--burst_length','-b' ,default=16, type=int, help='batch size')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='att_kpn_noise',
                        help='the checkpoint to eval')
    parser.add_argument('--model_type', '-m' ,default="attKPN", help='type of model : KPN, attKPN, attWKPN')
    parser.add_argument('--save_img', '-s' ,default="", type=str, help='save image in eval_img folder ')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')

    args = parser.parse_args()
    #

    test_wave(args)
