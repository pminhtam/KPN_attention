import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint
from utils.training_util import calculate_psnr, calculate_ssim
from utils.data_provider_DGF import *
from utils.KPN import KPN,LossFunc
from utils.KPN_DGF import KPN_DGF,Att_KPN_DGF,Att_Weight_KPN_DGF
from collections import OrderedDict
from torch.nn import functional as F

import torchvision.transforms as transforms
torch.manual_seed(0)
def eval(args):
    color = args.color
    print('Eval Process......')
    burst_length = 8
    # print(args.checkpoint)
    checkpoint_dir = "models/" + args.checkpoint
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # the path for saving eval images
    eval_dir = "eval_img"
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    # dataset and dataloader
    data_set = SingleLoader_DGF(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size,burst_length=burst_length)
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # model here
    if args.model_type == "attKPN":
        model = Att_KPN_DGF(
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
    elif args.model_type == "attWKPN":
        model = Att_Weight_KPN_DGF(
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
    if args.cuda:
        model = model.cuda()

    if args.mGPU:
        model = nn.DataParallel(model)
    # load trained model
    ckpt = load_checkpoint(checkpoint_dir,cuda=args.cuda)

    state_dict = ckpt['state_dict']
    if not args.cuda:
        new_state_dict = OrderedDict()
        if not args.cuda:
            for k, v in state_dict.items():
                name = k[0:]  # remove `module.`
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(ckpt['state_dict'])
    print('The model has been loaded from epoch {}, n_iter {}.'.format(ckpt['epoch'], ckpt['global_iter']))
    # switch the eval mode
    model.eval()

    # data_loader = iter(data_loader)
    trans = transforms.ToPILImage()

    with torch.no_grad():
        psnr = 0.0
        ssim = 0.0
        for i, (image_noise_hr,image_noise_lr, image_gt_hr) in enumerate(data_loader):
            if i < 100:
                # data = next(data_loader)
                if args.cuda:
                    burst_noise = image_noise_lr.cuda()
                    gt = image_gt_hr.cuda()
                else:
                    burst_noise = image_noise_lr
                    gt = image_gt_hr
                if color:
                    b, N, c, h, w = image_noise_lr.size()
                    feedData = image_noise_lr.view(b, -1, h, w)
                else:
                    feedData = image_noise_lr
                pred_i, pred = model(feedData, burst_noise[:, 0:burst_length, ...])
                gt = gt.unsqueeze(0)
                _, _, h_hr, w_hr = gt.size()
                _, _, h_lr, w_lr = pred.size()
                gt_down = F.interpolate(gt, (h_lr, w_lr), mode='bilinear', align_corners=True)
                pred_up = F.interpolate(pred, (h_hr, w_hr), mode='bilinear', align_corners=True)

                if not color:
                    psnr_t_up = calculate_psnr(pred_up, gt)
                    ssim_t_up = calculate_ssim(pred_up, gt)
                    psnr_t_down = calculate_psnr(pred, gt_down)
                    ssim_t_down = calculate_ssim(pred, gt_down)
                    print("UP   :  PSNR : ", str(psnr_t_up), " :  SSIM : ", str(ssim_t_up), " : DOWN   :  PSNR : ",
                          str(psnr_t_down), " :  SSIM : ", str(ssim_t_down))

                pred = torch.clamp(pred, 0.0, 1.0)

                if args.cuda:
                    pred = pred.cpu()
                    gt = gt.cpu()
                    burst_noise = burst_noise.cpu()

                trans(burst_noise[0, 0, ...].squeeze()).save(os.path.join(eval_dir, '{}_noisy.png'.format(i)), quality=100)
                trans(pred.squeeze()).save(os.path.join(eval_dir, '{}_pred_{:.2f}dB.png'.format(i, psnr_t_up)), quality=100)
                trans(gt.squeeze()).save(os.path.join(eval_dir, '{}_gt.png'.format(i)), quality=100)

                # print('{}-th image is OK, with PSNR: {:.2f}dB, SSIM: {:.4f}'.format(i, psnr_t, ssim_t))
            else:
                break
        # print('All images are OK, average PSNR: {:.2f}dB, SSIM: {:.4f}'.format(psnr/100, ssim/100))


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir', default='/home/dell/Downloads/noise', help='path to noise folder image')
    parser.add_argument('--gt_dir',default='/home/dell/Downloads/gt', help='path to gt folder image')
    parser.add_argument('--image_size',default=256, type=int, help='size of image')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--mGPU', '-m', action='store_true', help='whether to train on multiple GPUs')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='att_kpn_dgf',
                        help='the checkpoint to eval')
    parser.add_argument('--color',default=True, action='store_true')
    parser.add_argument('--model_type',default="KPN", help='type of model : KPN, attKPN, attWKPN')

    args = parser.parse_args()
    #
    eval(args)



