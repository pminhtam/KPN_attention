import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import shutil
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms
# import setproctitle
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint
from utils.training_util import calculate_psnr, calculate_ssim
from utils.data_provider_DGF import *
from utils.data_provider_DGF_synthetic import SingleLoader_DGF_synth
from utils.loss import LossBasic,WaveletLoss,tv_loss,LossAnneal_i,AlginLoss
from model.KPN_DGF import KPN_DGF,Att_KPN_DGF,Att_Weight_KPN_DGF,Att_KPN_Wavelet_DGF

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
    lr_step_size = 5
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
    if args.data_type == 'real':
        data_set = SingleLoader_DGF(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size,burst_length=burst_length)
    elif args.data_type == "synth":
        data_set = SingleLoader_DGF_synth(gt_dir=args.gt_dir,image_size=args.image_size,burst_length=burst_length)
    elif args.data_type == 'raw':
        data_set = SingleLoader_DGF_raw(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size,burst_length=args.burst_length)
    else:
        print("Wrong type data")
        return
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    # model here
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
            core_bias=False,
            in_channel=args.in_channel
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
            core_bias=False,
            in_channel=args.in_channel
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
            core_bias=False,
            in_channel=args.in_channel
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
            core_bias=False,
            in_channel=args.in_channel
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
    # loss_func = LossFunc(
    #     coeff_basic=1.0,
    #     coeff_anneal=1.0,
    #     gradient_L1=True,
    #     alpha=0.9998,
    #     beta=100.0
    # )
    loss_func = LossBasic()
    # loss_func = AlginLoss()
    loss_func_i = LossAnneal_i()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not restart_train:
        try:
            checkpoint = load_checkpoint(checkpoint_dir,cuda=device=='cuda',best_or_latest=args.load_type)
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
        for step, (image_noise_hr,image_noise_lr, image_gt_hr, image_gt_lr) in enumerate(data_loader):
            # print(burst_noise.size())
            # print(gt.size())
            burst_noise = image_noise_lr.to(device)
            gt = image_gt_hr.to(device)
            image_gt_lr = image_gt_lr.to(device)
            image_noise_hr = image_noise_hr.to(device)

            if color:
                b, N, c, h, w = image_noise_lr.size()
                feedData = image_noise_lr.view(b, -1, h, w).to(device)
            else:
                feedData = image_noise_lr
            # print('white_level', white_level, white_level.size())
            # print("feedData   : ",feedData.size())
            #
            pred_i, pred = model(feedData, burst_noise[:, 0:burst_length, ...],image_noise_hr)
            #
            # loss_basic, loss_anneal = loss_func(pred_i, pred, gt, global_step)
            loss_basic = loss_func(pred, gt)
            loss_i =loss_func_i(global_step, pred_i, image_gt_lr)
            loss = loss_basic + loss_i
            if args.wavelet_loss:
                loss_wave = loss_func2(pred,gt)
                # print(loss_wave)
                loss = loss_basic + loss_wave + loss_i
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
                # print("burst_noise  : ",burst_noise.size())
                # print("gt   :  ",gt.size())
                # print("feedData   : ", feedData.size())
                psnr = calculate_psnr(pred, gt)
                ssim = calculate_ssim(pred, gt)

                # add scalars to tensorboardX
                log_writer.add_scalar('loss_basic', loss_basic, global_step)
                # log_writer.add_scalar('loss_anneal', loss_anneal, global_step)
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
                print('Save   : {:-4d}\t| epoch {:2d}\t| step {:4d}\t| loss_basic: {:.4f}\t|'
                      ' loss: {:.4f}'
                      .format(global_step, epoch, step, loss_basic, loss))
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
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise', help='path to noise folder image')
    parser.add_argument('--gt_dir', '-g' , default='/home/dell/Downloads/gt', help='path to gt folder image')
    parser.add_argument('--data_type', '-dt' , default='real', help='real | synth | raw')
    parser.add_argument('--image_size', '-sz' , default=128, type=int, help='size of image')
    parser.add_argument('--epoch', '-e' ,default=1000, type=int, help='batch size')
    parser.add_argument('--batch_size','-bs' ,  default=2, type=int, help='batch size')
    parser.add_argument('--burst_length', '-b', default=16, type=int, help='batch size')
    parser.add_argument('--save_every','-se' , default=200, type=int, help='save_every')
    parser.add_argument('--loss_every', '-le' , default=10, type=int, help='loss_every')
    parser.add_argument('--restart','-r' ,  action='store_true', help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=2, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--mGPU', '-mg', action='store_true', help='whether to train on multiple GPUs')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='kpn',
                        help='the checkpoint to eval')
    parser.add_argument('--color','-cl' , default=True, action='store_true')
    parser.add_argument('--model_type','-m' ,default="KPN", help='type of model : KPN, attKPN, attWKPN, attKPN_Wave')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')
    parser.add_argument('--wavelet_loss','-wl' , default=False, action='store_true')
    parser.add_argument('--bn','-bn' , default=False, action='store_true', help='Use BatchNorm2d')
    parser.add_argument('--in_channel', '-ic', default=3,type=int, help='number of color input')

    args = parser.parse_args()

    train(args.num_workers,args.cuda, args.restart, args.mGPU)
