import argparse
import os
import torch
from utils.training_util import load_checkpoint
from utils.data_provider_DGF import pixel_unshuffle
from model.KPN import KPN
from model.Att_KPN import Att_KPN
from model.Att_Weight_KPN import Att_Weight_KPN
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image
import time
from torch.nn import functional as F
from utils.training_util import calculate_psnr, calculate_ssim

torch.manual_seed(0)
def load_data(image_file,burst_length):
    image_noise = transforms.ToTensor()(Image.open(image_file).convert('RGB'))
    image_noise = pixel_unshuffle(image_noise, 2)

    while len(image_noise) < burst_length:

        image_noise = torch.cat((image_noise,image_noise[-2:-1]),dim=0)
    if len(image_noise) > burst_length:
        image_noise = image_noise[0:8]
    image_noise_burst_crop = image_noise.unsqueeze(0)
    return image_noise_burst_crop

def test_multi(args):
    color = True
    burst_length = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "attKPN":
        model = Att_KPN(
            color=color,
            burst_length=burst_length,
            blind_est=True,
            kernel_size=[5],
            sep_conv=False,
            channel_att=True,
            spatial_att=True,
            upMode="bilinear",
            core_bias=False
        )
    elif args.model_type == "attWKPN":
        model = Att_Weight_KPN(
            color=color,
            burst_length=burst_length,
            blind_est=True,
            kernel_size=[5],
            sep_conv=False,
            channel_att=True,
            spatial_att=True,
            upMode="bilinear",
            core_bias=False
        )
    elif args.model_type == "KPN":
        model = KPN(
            color=color,
            burst_length=burst_length,
            blind_est=True,
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
    # model2 = KPN(
    #     color=color,
    #     burst_length=burst_length,
    #     blind_est=True,
    #     kernel_size=[5],
    #     sep_conv=False,
    #     channel_att=False,
    #     spatial_att=False,
    #     upMode="bilinear",
    #     core_bias=False
    # )
    checkpoint_dir = "checkpoints/" + args.checkpoint
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # load trained model
    ckpt = load_checkpoint(checkpoint_dir,cuda=device=='cuda')
    state_dict = ckpt['state_dict']
    # if not args.cuda:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # else:
    #     model.load_state_dict(ckpt['state_dict'])

    #############################################
    # checkpoint_dir = "checkpoints/" + "kpn"
    # if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
    #     print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # # load trained model
    # ckpt = load_checkpoint(checkpoint_dir,cuda=device=='cuda')
    # state_dict = ckpt['state_dict']
    # new_state_dict = OrderedDict()
    # if not args.cuda:
    #     for k, v in state_dict.items():
    #         name = k[7:]  # remove `module.`
    #         new_state_dict[name] = v
    # # model.load_state_dict(ckpt['state_dict'])
    # model2.load_state_dict(new_state_dict)
    ###########################################
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
        image_noise = load_data(noisy_path[i],burst_length)
        begin = time.time()
        image_noise_batch = image_noise.to(device)
        # print(image_noise.size())
        # print(image_noise_batch.size())
        burst_noise = image_noise_batch.to(device)
        if color:
            b, N, c, h, w = burst_noise.size()
            feedData = burst_noise.view(b, -1, h, w)
        else:
            feedData = burst_noise
        # print(feedData.size())
        pred_i, pred = model(feedData, burst_noise[:, 0:burst_length, ...])
        del pred_i
        # pred_i2, pred2 = model2(feedData, burst_noise[:, 0:burst_length, ...])
        # print("Time : ", time.time()-begin)
        pred = pred.detach().cpu()
        gt = transforms.ToTensor()(Image.open(clean_path[i]).convert('RGB'))
        # print(pred_i.size())
        # print(pred.size())
        # print(gt.size())
        gt = gt.unsqueeze(0)
        _, _, h_hr, w_hr = gt.size()
        _, _, h_lr, w_lr = pred.size()
        gt_down = F.interpolate(gt,(h_lr,w_lr), mode='bilinear', align_corners=True)
        pred_up = F.interpolate(pred,(h_hr,w_hr), mode='bilinear', align_corners=True)
        # print("After interpolate")
        # print(pred_up.size())
        # print(gt_down.size())
        psnr_t_up = calculate_psnr(pred_up, gt)
        ssim_t_up = calculate_ssim(pred_up, gt)
        psnr_t_down = calculate_psnr(pred, gt_down)
        ssim_t_down = calculate_ssim(pred, gt_down)
        print(i,"   UP   :  PSNR : ", str(psnr_t_up)," :  SSIM : ", str(ssim_t_up), " : DOWN   :  PSNR : ", str(psnr_t_down)," :  SSIM : ", str(ssim_t_down))

        if args.save_img != '':
            if not os.path.exists(args.save_img):
                os.makedirs(args.save_img)
            plt.figure(figsize=(15, 15))
            plt.imshow(np.array(trans(pred_up[0])))
            plt.title("denoise KPN split "+args.model_type, fontsize=25)
            image_name = noisy_path[i].split("/")[-1].split(".")[0]
            plt.axis("off")
            plt.suptitle(image_name+"   UP   :  PSNR : "+ str(psnr_t_up)+" :  SSIM : "+ str(ssim_t_up), fontsize=25)
            plt.savefig( os.path.join(args.save_img,image_name + "_" + args.checkpoint + '.png'),pad_inches=0)

        # print(np.array(trans(mf8[0])))
        """
        if args.save_img:
            plt.figure(figsize=(30, 9))
            plt.subplot(1,3,1)
            plt.imshow(np.array(trans(pred[0])))
            plt.title("denoise "+args.model_type, fontsize=26)
            # plt.subplot(1,3,2)
            # plt.imshow(np.array(trans(pred2[0])))
            # plt.title("denoise KPN")
            # plt.show()
            plt.subplot(1,3,2)
            plt.imshow(np.array(trans(gt[0])))
            plt.title("gt", fontsize=26)
            plt.subplot(1,3,3)
            plt.imshow(np.array(trans(image_noise[0][1])))
            plt.title("noise ", fontsize=26)
            plt.suptitle(str(i)+"   UP   :  PSNR : "+ str(psnr_t_up)+" :  SSIM : "+ str(ssim_t_up)+ " : DOWN   :  PSNR : "+ str(psnr_t_down)+" :  SSIM : "+ str(ssim_t_down), fontsize=26)
            plt.savefig("checkpoints/"+ args.model_type+str(i)+'.png',pad_inches=0)
        """
        # plt.show()

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/FullTest/noisy', help='path to noise image file')
    parser.add_argument('--gt','-g', default='/home/dell/Downloads/FullTest/clean', help='path to noise image file')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='att_kpn',
                        help='the checkpoint to eval')
    parser.add_argument('--model_type', '-m' , default="attKPN", help='type of model : KPN, attKPN, attWKPN')
    parser.add_argument('--save_img', '-s'  , default="", type=str, help='save image in eval_img folder ')

    args = parser.parse_args()
    #

    test_multi(args)



