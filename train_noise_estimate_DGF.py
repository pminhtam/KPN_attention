import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse

import os
import time
import shutil

from tensorboardX import SummaryWriter
# import setproctitle
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint
from utils.training_util import calculate_psnr, calculate_ssim
from utils.data_provider_DGF import SingleLoader_DGF
from utils.loss import LossBasic,WaveletLoss
from model.KPN_noise_estimate_DGF import KPN_noise_DGF,Att_KPN_noise_DGF,Att_Weight_KPN_noise_DGF


def train(num_workers, cuda, restart_train, mGPU):
    # torch.set_num_threads(num_threads)

    color = True
    batch_size = args.batch_size
    lr = 2e-4
    lr_decay = 0.89125093813
    n_epoch = args.epoch
    # num_workers = 8
    save_freq = args.save_every
    loss_freq = args.loss_every
    lr_step_size = 100
    burst_length = args.burst_length
    # checkpoint path
    checkpoint_dir = "checkpoints/" + args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # logs path
    logs_dir = "checkpoints/logs/" + args.checkpoint
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    shutil.rmtree(logs_dir)
    log_writer = SummaryWriter(logs_dir)

    # dataset and dataloader
    data_set = SingleLoader_DGF(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size,burst_length=burst_length)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    # model here
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
    elif args.model_type == 'KPN':
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
    if cuda:
        model = model.cuda()

    if mGPU:
        model = nn.DataParallel(model)
    model.train()

    # loss function here
    loss_func = LossBasic()
    if args.wavelet_loss:
        print("Use wavelet loss")
        loss_func2 = WaveletLoss()
    # Optimizer here
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    optimizer.zero_grad()

    # learning rate scheduler here
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay)

    average_loss = MovingAverage(save_freq)
    if not restart_train:
        try:
            checkpoint = load_checkpoint(checkpoint_dir,cuda , best_or_latest=args.load_type)
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    else:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        if os.path.exists(checkpoint_dir):
            pass
            # files = os.listdir(checkpoint_dir)
            # for f in files:
            #     os.remove(os.path.join(checkpoint_dir, f))
        else:
            os.mkdir(checkpoint_dir)
        print('=> training')


    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        # decay the learning rate

        # print('='*20, 'lr={}'.format([param['lr'] for param in optimizer.param_groups]), '='*20)
        t1 = time.time()
        for step, (image_noise_hr,image_noise_lr, image_gt_hr, _) in enumerate(data_loader):
            # print(burst_noise.size())
            # print(gt.size())
            if cuda:
                burst_noise = image_noise_lr.cuda()
                gt = image_gt_hr.cuda()
                image_noise_hr = image_noise_hr.cuda()
                noise_gt = (image_noise_hr-image_gt_hr).cuda()
            else:
                burst_noise = image_noise_lr
                gt = image_gt_hr
                noise_gt = image_noise_hr - image_gt_hr
            #
            _, pred,noise = model(burst_noise,image_noise_hr)
            # print(pred.size())
            #
            loss_basic = loss_func(pred, gt)
            loss_noise = loss_func(noise,noise_gt)
            loss = loss_basic + loss_noise
            if args.wavelet_loss:
                loss_wave = loss_func2(pred,gt)
                loss_wave_noise = loss_func2(noise,noise_gt)
                # print(loss_wave)
                loss = loss_basic + loss_wave + loss_noise + loss_wave_noise
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update the average loss
            average_loss.update(loss)
            # global_step

            if not color:
                pred = pred.unsqueeze(1)
                gt = gt.unsqueeze(1)
            if global_step %loss_freq ==0:
                # calculate PSNR
                print("burst_noise  : ",burst_noise.size())
                print("gt   :  ",gt.size())
                psnr = calculate_psnr(pred, gt)
                ssim = calculate_ssim(pred, gt)

                # add scalars to tensorboardX
                log_writer.add_scalar('loss_basic', loss_basic, global_step)
                log_writer.add_scalar('loss_total', loss, global_step)
                log_writer.add_scalar('psnr', psnr, global_step)
                log_writer.add_scalar('ssim', ssim, global_step)

                # print
                print('{:-4d}\t| epoch {:2d}\t| step {:4d}\t| loss_basic: {:.4f}\t|'
                      ' loss: {:.4f}\t| PSNR: {:.2f}dB\t| SSIM: {:.4f}\t| time:{:.2f} seconds.'
                      .format(global_step, epoch, step, loss_basic, loss, psnr, ssim, time.time()-t1))
                t1 = time.time()


            if global_step % save_freq == 0:
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False

                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict()
                }
                save_checkpoint(
                    save_dict, is_best, checkpoint_dir, global_step, max_keep=10
                )
            global_step += 1
        print('Epoch {} is finished, time elapsed {:.2f} seconds.'.format(epoch, time.time()-epoch_start_time))
        lr_cur = [param['lr'] for param in optimizer.param_groups]
        if lr_cur[0] > 5e-6:
            scheduler.step()
        else:
            for param in optimizer.param_groups:
                param['lr'] = 5e-6


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n',  default='/home/dell/Downloads/noise', help='path to noise folder image')
    parser.add_argument('--gt_dir','-g' , default='/home/dell/Downloads/gt', help='path to gt folder image')
    parser.add_argument('--image_size', '-sz' , default=128, type=int, help='size of image')
    parser.add_argument('--batch_size',  '-bs' , default=1, type=int, help='batch size')
    parser.add_argument('--burst_length', '-b', default=1, type=int, help='batch size')
    parser.add_argument('--epoch', '-e' ,default=1000, type=int, help='batch size')
    parser.add_argument('--save_every', '-se' , default=200, type=int, help='save_every')
    parser.add_argument('--loss_every', '-le' ,default=100, type=int, help='loss_every')
    parser.add_argument('--restart',  '-r' ,  action='store_true', help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--mGPU', '-mg', action='store_true', help='whether to train on multiple GPUs')
    parser.add_argument('--eval', action='store_true', help='whether to work on the evaluation mode')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='kpn_noise_dgf',
                        help='the checkpoint to eval')
    parser.add_argument('--color','-cl' , default=True, action='store_true')
    parser.add_argument('--model_type', '-m' , default="KPN", help='type of model : KPN, attKPN, attWKPN')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')
    parser.add_argument('--wavelet_loss','-wl' , default=False, action='store_true')

    args = parser.parse_args()
    #
    if args.eval:
        pass
    else:
        train(args.num_workers,args.cuda, args.restart, args.mGPU)
