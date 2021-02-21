import argparse
from utils.data_provider import *

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from PIL import Image
import scipy.io
torch.manual_seed(0)

def test_multi(args):
    # model= save_dict['state_dict']
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    # all_clean_imgs = scipy.io.loadmat(args.gt)['ValidationGtBlocksSrgb']
    all_clean_imgs = scipy.io.loadmat(args.gt)['ValidationNoisyBlocksSrgb']
    i_imgs,i_blocks, _,_,_ = all_clean_imgs.shape

    for i_img in range(i_imgs):
        for i_block in range(i_blocks):
            gt = transforms.ToTensor()(Image.fromarray(all_clean_imgs[i_img][i_block]))
            gt = gt.unsqueeze(0)
            # print(pred_i.size())
            # print(pred[0].size())
            if args.save_img != '':
                if not os.path.exists(args.save_img):
                    os.makedirs(args.save_img)
                plt.figure(figsize=(15, 15))
                plt.imshow(np.array(trans(gt[0])))
                plt.title("NOISY ", fontsize=25)
                image_name = str(i_img) + "_" + str(i_block)
                plt.axis("off")
                plt.savefig( os.path.join(args.save_img,image_name + '.png'),pad_inches=0)

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--gt','-g', default='data/ValidationNoisyBlocksSrgb.mat', help='path to noise image file')
    parser.add_argument('--save_img', "-s" ,default="img/validate_noise", type=str, help='save image in eval_img folder ')

    args = parser.parse_args()
    #

    test_multi(args)
