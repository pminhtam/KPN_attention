import argparse
import os
import torch
from utils.training_util import load_checkpoint
from utils.data_provider_DGF import pixel_unshuffle
from model.KPN_noise_estimate_DGF import KPN_noise_DGF,Att_KPN_noise_DGF,Att_Weight_KPN_noise_DGF
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image
import time
from utils.training_util import calculate_psnr, calculate_ssim
import math


torch.manual_seed(0)
def load_data(image_file,burst_length):
    image_noise = transforms.ToTensor()(Image.open(image_file).convert('RGB'))
    image_noise_hr = image_noise
    upscale_factor = int(math.sqrt(burst_length))
    image_noise = pixel_unshuffle(image_noise, upscale_factor)
    while len(image_noise) < burst_length:
        image_noise = torch.cat((image_noise,image_noise[-2:-1]),dim=0)
    if len(image_noise) > burst_length:
        image_noise = image_noise[0:8]
    image_noise_burst_crop = image_noise.unsqueeze(0)
    return image_noise_burst_crop,image_noise_hr.unsqueeze(0)

def test_multi(args):
    color = True
    burst_length = args.burst_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "attKPN":
        model = Att_KPN_noise_DGF(
            color=color,
            burst_length=burst_length,
            blind_est=False,
            kernel_size=[5],
            sep_conv=False,
            channel_att=True,
            spatial_att=True,
            upMode="bilinear",
            core_bias=False
        )
    elif args.model_type == "attWKPN":
        model = Att_Weight_KPN_noise_DGF(
            color=color,
            burst_length=burst_length,
            blind_est=False,
            kernel_size=[5],
            sep_conv=False,
            channel_att=True,
            spatial_att=True,
            upMode="bilinear",
            core_bias=False
        )
    elif args.model_type == "KPN":
        model = KPN_noise_DGF(
            color=color,
            burst_length=burst_length,
            blind_est=False,
            kernel_size=[5],
            sep_conv=False,
            channel_att=False,
            spatial_att=False,
            upMode="bilinear",
            core_bias=False
        )
    else:
        print(" Model type not valid")
        return

    checkpoint_dir = "checkpoints/" + args.checkpoint
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # load trained model
    ckpt = load_checkpoint(checkpoint_dir,cuda=device=='cuda',best_or_latest=args.load_type)
    state_dict = ckpt['state_dict']
    # if not args.cuda:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    print('The model has been loaded from epoch {}, n_iter {}.'.format(ckpt['epoch'], ckpt['global_iter']))
    # switch the eval mode
    model.to(device)
    model.eval()
    # model2.eval()
    # model= save_dict['state_dict']
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    clean_path = [ i.replace("noisy","clean") for i in noisy_path]
    for i in range(len(noisy_path)):
        image_noise,image_noise_hr = load_data(noisy_path[i],burst_length)
        begin = time.time()
        image_noise_batch = image_noise.to(device)
        burst_noise = image_noise_batch.to(device)
        image_noise_hr = image_noise_hr.to(device)

        pred = model(burst_noise,image_noise_hr)
        pred = pred.detach().cpu()
        # print("Time : ", time.time()-begin)
        gt = transforms.ToTensor()(Image.open(clean_path[i]).convert('RGB'))
        gt = gt.unsqueeze(0)
        # print(pred_i.size())
        # print(pred[0].size())
        psnr_t = calculate_psnr(pred, gt)
        ssim_t = calculate_ssim(pred, gt)
        print(i, "   UP   :  PSNR : ", str(psnr_t), " :  SSIM : ", str(ssim_t))
        if args.save_img != '':
            if not os.path.exists(args.save_img):
                os.makedirs(args.save_img)
            plt.figure(figsize=(15, 15))
            plt.imshow(np.array(trans(pred[0])))
            plt.title("denoise KPN noise DGF " + args.model_type, fontsize=25)
            image_name = noisy_path[i].split("/")[-1].split(".")[0]
            plt.axis("off")
            plt.suptitle(image_name + "   UP   :  PSNR : " + str(psnr_t) + " :  SSIM : " + str(ssim_t), fontsize=25)
            plt.savefig(os.path.join(args.save_img, image_name + "_" + args.checkpoint + '.png'), pad_inches=0)

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

    test_multi(args)



