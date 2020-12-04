import argparse
from utils.training_util import load_checkpoint
from utils.data_provider import *
from model.KPN import KPN
from model.Att_KPN import Att_KPN
from model.Att_Weight_KPN import Att_Weight_KPN
from collections import OrderedDict
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import glob
from PIL import Image
import time
torch.manual_seed(0)

def load_data(dir,image_size,burst_length):
    image_files = sorted(glob.glob(dir + "/*"))[:8]
    print(image_files)
    image_0 = Image.open(image_files[0]).convert('RGB')
    w = image_size
    h = image_size
    # print(image_0.size[-1])
    nw = image_0.size[0] - w
    nh = image_0.size[1] - h
    print(nw,nh)
    if nw < 0 or nh < 0:
        raise RuntimeError("Image is to small {} for the desired size {}". \
                           format((image_0.size(-1), image_0.size(-2)), (w, h))
                           )
    idx_w = torch.randint(0, nw + 1, (1,))[0]
    idx_h = torch.randint(0, nh + 1, (1,))[0]
    print(idx_w,idx_h)
    image_noise = [transforms.ToTensor()(Image.open(img_path).convert('RGB'))[:, idx_h:(idx_h + h), idx_w:(idx_w + w)] for
               img_path in image_files]
    while len(image_noise) < burst_length:
        image_noise.append(image_noise[-1])
    image_noise_burst_crop = torch.stack(image_noise, dim=0)
    image_noise_burst_crop = image_noise_burst_crop.unsqueeze(0)
    print("image_noise_burst_crop shape : ",image_noise_burst_crop.size())
    return image_noise_burst_crop

def test_multi(dir,image_size,args):
    num_workers = 1
    batch_size = 1
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
    model2 = KPN(
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
    checkpoint_dir = "checkpoints/" + args.checkpoint
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # load trained model
    ckpt = load_checkpoint(checkpoint_dir,cuda=device=='cuda')
    state_dict = ckpt['state_dict']
    new_state_dict = OrderedDict()
    # if not args.mGPU:
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # model.load_state_dict(ckpt['state_dict'])
    model.load_state_dict(new_state_dict)

    checkpoint_dir = "checkpoints/" + "kpn"
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    # load trained model
    ckpt = load_checkpoint(checkpoint_dir,cuda=device=='cuda')
    state_dict = ckpt['state_dict']
    if not args.cuda:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model2.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(ckpt['state_dict'])
    print('The model has been loaded from epoch {}, n_iter {}.'.format(ckpt['epoch'], ckpt['global_iter']))
    # switch the eval mode
    model.to(device)
    model2.to(device)
    model.eval()
    model2.eval()
    # model= save_dict['state_dict']
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    for i in range(10):
        image_noise = load_data(dir,image_size,burst_length)
        begin = time.time()
        image_noise_batch = image_noise.to(device)
        print(image_noise_batch.size())
        burst_size = image_noise_batch.size()[1]
        burst_noise = image_noise_batch.to(device)
        if color:
            b, N, c, h, w = burst_noise.size()
            feedData = burst_noise.view(b, -1, h, w)
        else:
            feedData = burst_noise
        # print(feedData.size())
        pred_i, pred = model(feedData, burst_noise[:, 0:burst_length, ...])
        pred_i2, pred2 = model2(feedData, burst_noise[:, 0:burst_length, ...])
        pred = pred.detach().cpu()
        pred2 = pred2.detach().cpu()
        print("Time : ", time.time()-begin)
        print(pred_i.size())
        print(pred.size())
        if args.save_img != '':
            # print(np.array(trans(mf8[0])))
            plt.figure(figsize=(10, 3))
            plt.subplot(1,3,1)
            plt.imshow(np.array(trans(pred[0])))
            plt.title("denoise attKPN")
            plt.subplot(1,3,2)
            plt.imshow(np.array(trans(pred2[0])))
            plt.title("denoise KPN")
            # plt.show()
            plt.subplot(1,3,3)
            plt.imshow(np.array(trans(image_noise[0][0])))
            plt.title("noise ")
            image_name = str(i)
            plt.savefig(os.path.join(args.save_img,image_name + "_" + args.checkpoint + '.png'),pad_inches=0)
            # plt.show()

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/samples/samples', help='path to noise image file')
    # parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise/0001_NOISY_SRGB', help='path to noise image file')
    parser.add_argument('--image_size','-sz' , type=int,default=256, help='size of image')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='attwkpn',
                        help='the checkpoint to eval')
    parser.add_argument('--model_type',default="attWKPN", help='type of model : KPN, attKPN, attWKPN')
    parser.add_argument('--save_img', "-s" ,default="", type=str, help='save image in eval_img folder ')

    args = parser.parse_args()
    #

    test_multi(args.noise_dir,args.image_size,args)



