import argparse
from utils.training_util import load_checkpoint
from utils.data_provider import *
from model.KPN_DGF import KPN_DGF,Att_KPN_DGF,Att_Weight_KPN_DGF,Att_KPN_Wavelet_DGF

from collections import OrderedDict
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import glob
from PIL import Image
import time
import math
from utils.training_util import calculate_psnr, calculate_ssim
import scipy.io
import scipy.io as sio

torch.manual_seed(0)
def load_data(image_noise,burst_length):
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
        model = Att_KPN_DGF(
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
    elif args.model_type == "attKPN_Wave":
        model = Att_KPN_Wavelet_DGF(
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
        model = Att_Weight_KPN_DGF(
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
        model = KPN_DGF(
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
    model.load_state_dict(state_dict)
    # else:
    #     model.load_state_dict(ckpt['state_dict'])

    model.to(device)
    print('The model has been loaded from epoch {}, n_iter {}.'.format(ckpt['epoch'], ckpt['global_iter']))
    # switch the eval mode
    noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    model.eval()
    torch.manual_seed(0)
    trans = transforms.ToPILImage()
    if not os.path.exists(args.save_img):
        os.makedirs(args.save_img)
    for i in range(len(noisy_path)):
        image_noise = transforms.ToTensor()(Image.open(noisy_path[i]).convert('RGB'))
        image_noise,image_noise_hr = load_data(image_noise,burst_length)
        image_noise_hr = image_noise_hr.to(device)
        # begin = time.time()
        image_noise_batch = image_noise.to(device)
        # print(image_noise_batch.size())
        # burst_size = image_noise_batch.size()[1]
        burst_noise = image_noise_batch.to(device)
        # print(burst_noise.size())
        # print(image_noise_hr.size())
        if color:
            b, N, c, h, w = burst_noise.size()
            feedData = burst_noise.view(b, -1, h, w)
        else:
            feedData = burst_noise
        pred_i, pred = model(feedData, burst_noise[:, 0:burst_length, ...],image_noise_hr)
        del pred_i
        print(pred.size())
        pred = np.array(trans(pred[0].cpu()))
        print(pred.shape)
        if args.save_img != '':
            if not os.path.exists(args.save_img):
                os.makedirs(args.save_img)
            # mat_contents['image'] = pred
            # print(mat_contents)
            print("save : ", os.path.join(args.save_img,noisy_path[i].split("/")[-1].split(".")[0]+'.mat'))
            data = {"Idenoised_crop": pred}
            # print(data)
            sio.savemat(os.path.join(args.save_img,noisy_path[i].split("/")[-1].split(".")[0]+'.mat'), data)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/FullTest/test/', help='path to noise image file')
    parser.add_argument('--burst_length','-b' ,default=4, type=int, help='batch size')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='att_kpn_dgf_4_new',
                        help='the checkpoint to eval')
    parser.add_argument('--model_type','-m' ,default="attKPN", help='type of model : KPN, attKPN, attWKPN , attKPN_Wave')
    parser.add_argument('--save_img', "-s" ,default="/home/dell/Downloads/FullTest/test_re", type=str, help='save image in eval_img folder ')
    parser.add_argument('--load_type', "-l" ,default="lastest", type=str, help='Load type best_or_latest ')

    args = parser.parse_args()
    #

    test_multi(args)
    # gt_file = "data/ValidationGtBlocksSrgb.mat"
    # mat = scipy.io.loadmat(gt_file)['ValidationGtBlocksSrgb']
    # print(mat.shape)
    # img = Image.fromarray(mat[0][1], 'RGB')
    # gt = transforms.ToTensor()(img)
    # print(gt)
