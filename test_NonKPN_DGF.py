import argparse
from utils.training_util import load_checkpoint
from utils.data_provider import *
from model.NonKPN_DGF import Att_NonKPN_Wavelet_DGF

from collections import OrderedDict
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import glob
from PIL import Image
import time
import math
from utils.training_util import calculate_psnr, calculate_ssim

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
    if  args.model_type == "attNonKPN_Wave":
        model = Att_NonKPN_Wavelet_DGF(
            color=color,
            burst_length=burst_length,
            blind_est=True,
            kernel_size=[3],
            sep_conv=False,
            channel_att=True,
            spatial_att=True,
            upMode="bilinear",
            core_bias=False,
            bn=args.bn
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
    # else:
    #     model.load_state_dict(ckpt['state_dict'])

    model.to(device)
    print('The model has been loaded from epoch {}, n_iter {}.'.format(ckpt['epoch'], ckpt['global_iter']))
    # switch the eval mode
    model.eval()
    # model= save_dict['state_dict']
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    clean_path = [ i.replace("noisy","clean") for i in noisy_path]
    for i in range(len(noisy_path)):
        image_noise,image_noise_hr = load_data(noisy_path[i],burst_length)
        image_noise_hr = image_noise_hr.to(device)
        # begin = time.time()
        image_noise_batch = image_noise.to(device)
        # print(image_noise_batch.size())
        burst_size = image_noise_batch.size()[1]
        burst_noise = image_noise_batch.to(device)
        # print(burst_noise.size())
        # print(image_noise_hr.size())
        if color:
            b, N, c, h, w = burst_noise.size()
            feedData = burst_noise.view(b, -1, h, w)
        else:
            feedData = burst_noise
        # print(feedData.size())
        pred = model(feedData, burst_noise[:, 0:burst_length, ...],image_noise_hr)
        pred = pred.detach().cpu()
        # print("Time : ", time.time()-begin)
        gt = transforms.ToTensor()(Image.open(clean_path[i]).convert('RGB'))
        gt = gt.unsqueeze(0)
        # print(pred_i.size())
        # print(pred[0].size())
        psnr_t = calculate_psnr(pred, gt)
        ssim_t = calculate_ssim(pred, gt)
        print(i,"   UP   :  PSNR : ", str(psnr_t)," :  SSIM : ", str(ssim_t))
        if args.save_img != '':
            if not os.path.exists(args.save_img):
                os.makedirs(args.save_img)
            plt.figure(figsize=(15, 15))
            plt.imshow(np.array(trans(pred[0])))
            plt.title("denoise KPN DGF "+args.model_type, fontsize=25)
            image_name = noisy_path[i].split("/")[-1].split(".")[0]
            plt.axis("off")
            plt.suptitle(image_name+"   UP   :  PSNR : "+ str(psnr_t)+" :  SSIM : "+ str(ssim_t), fontsize=25)
            plt.savefig( os.path.join(args.save_img,image_name + "_" + args.checkpoint + '.png'),pad_inches=0)
        """
        if args.save_img:
            # print(np.array(trans(mf8[0])))
            plt.figure(figsize=(30, 9))
            plt.subplot(1,3,1)
            plt.imshow(np.array(trans(pred[0])))
            plt.title("denoise DGF "+args.model_type, fontsize=26)
            plt.subplot(1,3,2)
            plt.imshow(np.array(trans(gt[0])))
            plt.title("gt ", fontsize=26)
            plt.subplot(1,3,3)
            plt.imshow(np.array(trans(image_noise_hr[0])))
            plt.title("noise ", fontsize=26)
            plt.axis("off")
            plt.suptitle(str(i)+"   UP   :  PSNR : "+ str(psnr_t)+" :  SSIM : "+ str(ssim_t), fontsize=26)
            plt.savefig("checkpoints/22_DGF_" + args.checkpoint+str(i)+'.png',pad_inches=0)
        """

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/FullTest/noisy', help='path to noise image file')
    parser.add_argument('--gt','-g', default='/home/dell/Downloads/FullTest/clean', help='path to noise image file')
    parser.add_argument('--burst_length','-b' ,default=16, type=int, help='batch size')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='att_kpn_dgf',
                        help='the checkpoint to eval')
    parser.add_argument('--model_type','-m' ,default="attKPN", help='type of model : attNonKPN_Wave')
    parser.add_argument('--save_img', "-s" ,default="", type=str, help='save image in eval_img folder ')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')
    parser.add_argument('--bn','-bn' , default=False, action='store_true', help='Use BatchNorm2d')

    args = parser.parse_args()
    #

    test_multi(args)



