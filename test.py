import argparse
import os
from model import *
from metric import *
from data_loader import SingleLoader,MultiLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from metric import  psnr
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint
from utils.training_util import calculate_psnr, calculate_ssim
from utils.data_provider import *
from utils.KPN import KPN,LossFunc
from utils.Att_KPN import Att_KPN
from utils.Att_Weight_KPN import Att_Weight_KPN

import torchvision.transforms as transforms
def test_multi(noise_dir,gt_dir,image_size,num_workers,checkpoint,resume):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiLoader(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=num_workers)
    model_single = SFD_C().to(device)
    model = MFD_C(model_single).to(device)
    if resume != '':
        print(device)
        save_dict = torch.load(os.path.join(checkpoint, resume), map_location=torch.device('cpu'))

        # if device == "cpu":
        #     save_dict = torch.load(os.path.join(checkpoint,resume),map_location=torch.device('cpu'))
        # else:
        #     save_dict = torch.load(os.path.join(checkpoint, resume))
        model.load_state_dict(save_dict['state_dict'])
    model.eval()
    trans = transforms.ToPILImage()
    for i in range(10):
        for step, (image_noise, image_gt) in enumerate(data_loader):

            image_noise_batch = image_noise.to(device)
            image_gt = image_gt.to(device)
            # print(image_noise_batch.size())
            burst_size = image_noise_batch.size()[0]
            mfinit1, mfinit2, mfinit3,mfinit4,mfinit5,mfinit6,mfinit7 = torch.zeros(7, 1, 64, image_size, image_size).to(device)
            mfinit8 = torch.zeros(1, 3, image_size, image_size).to(device)
            i = 0
            for i_burst in range(burst_size):
                frame = image_noise_batch[:,i_burst,:,:,:]
                # print(frame.size())
                if i == 0:
                    i += 1
                    dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8 = model(
                        frame, mfinit1, mfinit2, mfinit3, mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
                else:
                    dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8= model(dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8)
            # # print(np.array(trans(mf8[0])))
            # print(np.array(trans(dframe[0])).shape)
            # print(np.array(trans(image_gt[0])).shape)
            # plt.imshow(np.array(trans(dframe[0])))
            # plt.show()
            # plt.imshow(np.array(trans(image_gt[0])))
            # plt.show()
            print(psnr(np.array(trans(dframe[0])),np.array(trans(image_gt[0]))))
def eval(args):
    color = True
    print('Eval Process......')
    burst_length = 8

    checkpoint_dir = "mode" +  args.checkpoint
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # the path for saving eval images
    eval_dir = "eval_img"
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    files = os.listdir(eval_dir)
    for f in files:
        os.remove(os.path.join(eval_dir, f))

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
        model = Att_KPN(
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
        model = Att_Weight_KPN(
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
    if args.cuda:
        model = model.cuda()

    if args.mGPU:
        model = nn.DataParallel(model)
    # load trained model
    ckpt = load_checkpoint(checkpoint_dir)
    model.load_state_dict(ckpt['state_dict'])
    print('The model has been loaded from epoch {}, n_iter {}.'.format(ckpt['epoch'], ckpt['global_iter']))
    # switch the eval mode
    model.eval()

    # data_loader = iter(data_loader)
    trans = transforms.ToPILImage()

    with torch.no_grad():
        psnr = 0.0
        ssim = 0.0
        for i, (burst_noise, gt, white_level) in enumerate(data_loader):
            if i < 100:
                # data = next(data_loader)
                if args.cuda:
                    burst_noise = burst_noise.cuda()
                    gt = gt.cuda()
                    white_level = white_level.cuda()

                pred_i, pred = model(burst_noise, burst_noise[:, 0:burst_length, ...], white_level)

                burst_noise = burst_noise / white_level

                if not color:
                    psnr_t = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
                    ssim_t = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
                    psnr_noisy = calculate_psnr(burst_noise[:, 0, ...].unsqueeze(1), gt.unsqueeze(1))
                else:
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

                trans(burst_noise[0, 0, ...].squeeze()).save(os.path.join(eval_dir, '{}_noisy_{:.2f}dB.png'.format(i, psnr_noisy)), quality=100)
                trans(pred.squeeze()).save(os.path.join(eval_dir, '{}_pred_{:.2f}dB.png'.format(i, psnr_t)), quality=100)
                trans(gt.squeeze()).save(os.path.join(eval_dir, '{}_gt.png'.format(i)), quality=100)

                print('{}-th image is OK, with PSNR: {:.2f}dB, SSIM: {:.4f}'.format(i, psnr_t, ssim_t))
            else:
                break
        print('All images are OK, average PSNR: {:.2f}dB, SSIM: {:.4f}'.format(psnr/100, ssim/100))


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise', help='path to noise image file')
    parser.add_argument('--gt_dir','-g',  default='/home/dell/Downloads/gt', help='path to groud true image file')
    parser.add_argument('--image_size','-sz' , type=int,default=128, help='size of image')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint',
                        help='the folder checkpoint to save')
    parser.add_argument('--resume', '-r', type=str, default="MFD_C_47.pth.tar",
                        help='file name of checkpoint')
    parser.add_argument('--model_type', '-t', type=str, default='multi',help='type model train is single or multi')
    args = parser.parse_args()
    #
    test_multi(args.noise_dir,args.gt_dir,args.image_size,args.num_workers,args.checkpoint,args.resume)



