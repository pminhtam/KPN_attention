import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint
from utils.training_util import calculate_psnr, calculate_ssim
from utils.data_provider import *
from utils.KPN_noise_estimate import KPN_noise,Att_KPN_noise,Att_Weight_KPN_noise

from collections import OrderedDict

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
    data_set = MultiLoader(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size)
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # model here
    if args.model_type == "attKPN":
        model = Att_KPN_noise(
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
    elif args.model_type == "attWKPN":
        model = Att_Weight_KPN_noise(
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
    elif args.model_type == "KPN":
        model = KPN_noise(
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
    if args.cuda:
        model = model.cuda()

    if args.mGPU:
        model = nn.DataParallel(model)
    # load trained model
    ckpt = load_checkpoint(checkpoint_dir,cuda=args.cuda)

    state_dict = ckpt['state_dict']
    if not args.mGPU:
        new_state_dict = OrderedDict()
        if not args.cuda:
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(ckpt['state_dict'])
    print('The model has been loaded from epoch {}, n_iter {}.'.format(ckpt['epoch'], ckpt['global_iter']))
    # torch.save(model.state_dict(), "model_state.pth")
    # exit(0)
    # switch the eval mode
    model.eval()

    # data_loader = iter(data_loader)
    trans = transforms.ToPILImage()

    with torch.no_grad():
        psnr = 0.0
        ssim = 0.0
        torch.manual_seed(0)
        for i, (burst_noise, gt) in enumerate(data_loader):
            if i < 100:
                # data = next(data_loader)
                if args.cuda:
                    burst_noise = burst_noise.cuda()
                    gt = gt.cuda()
                pred_i, pred = model(burst_noise)



                psnr_t = calculate_psnr(pred, gt)
                ssim_t = calculate_ssim(pred, gt)
                psnr_noisy = calculate_psnr(burst_noise[:, 0, ...], gt)

                psnr += psnr_t
                ssim += ssim_t

                pred = torch.clamp(pred, 0.0, 1.0)

                if args.cuda:
                    pred = pred.cpu()
                    gt = gt.cpu()
                    burst_noise = burst_noise.cpu()
                if args.save_img:
                    trans(burst_noise[0, 0, ...].squeeze()).save(os.path.join(eval_dir, '{}_noisy_{:.2f}dB.png'.format(i, psnr_noisy)), quality=100)
                    trans(pred.squeeze()).save(os.path.join(eval_dir, '{}_pred_{:.2f}dB.png'.format(i, psnr_t)), quality=100)
                    trans(gt.squeeze()).save(os.path.join(eval_dir, '{}_gt.png'.format(i)), quality=100)

                print('{}-th image is OK, with PSNR: {:.2f} , SSIM: {:.4f}'.format(i, psnr_t, ssim_t))
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
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='kpn',
                        help='the checkpoint to eval')
    parser.add_argument('--color',default=True, action='store_true')
    parser.add_argument('--model_type',default="KPN", help='type of model : KPN, attKPN, attWKPN')
    parser.add_argument('--save_img',default=False, action='store_true', help='save image in eval_img folder ')

    args = parser.parse_args()
    #
    eval(args)



