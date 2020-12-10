import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import glob
import torchvision.transforms as transforms
import math
from utils.data_transform import random_flip, random_rotate
from utils.data_provider_DGF import random_cut,pixel_unshuffle
##

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
class SingleLoader_DGF_synth(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, gt_dir,image_size=128,burst_length=16):

        self.gt_dir = gt_dir
        self.image_size = image_size
        self.burst_length = burst_length
        self.upscale_factor = int(math.sqrt(self.burst_length))
        self.gt_path = []
        for files_ext in IMG_EXTENSIONS:
            self.gt_path.extend(glob.glob(self.gt_dir +"/**/*" + files_ext,recursive=True))
        
        if len(self.gt_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + self.gt_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        

        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.sigma_plus = 0.1
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue 
        """
        rand_hflip = torch.rand(1)[0]
        rand_vflip = torch.rand(1)[0]
        rand_affine = torch.rand(1)[0]
        angle = torch.randint(low= -20,high=20,size=(1,))[0]
        image_gt = Image.open(self.gt_path[index]).convert('RGB')
        # print(image_gt.size)
        nw = image_gt.size[-1] - self.image_size
        nh = image_gt.size[-2] - self.image_size
        while nw < 0 or nh < 0:
            print("del  : ",index,'  with size :', image_gt.size)
            del self.gt_path[index]
            if index >= len(self.gt_path):
                index = len(self.gt_path) -1
            image_gt = Image.open(self.gt_path[index]).convert('RGB')
            nw = image_gt.size[-1] - self.image_size
            nh = image_gt.size[-2] - self.image_size
        image_gt = random_flip(image_gt,rand_hflip,rand_vflip)
        image_gt = random_rotate(image_gt,rand_affine,angle)

        image_gt = self.transforms(image_gt)
        type_rand = torch.rand(1)
        if type_rand < 0.8:
            noise = torch.randn(image_gt.size())*self.sigma_plus*torch.rand(1)
        elif type_rand > 0.8 and type_rand < 1:
            noise = torch.rand(image_gt.size())*self.sigma_plus*torch.rand(1)
        image_noise = image_gt + noise
        image_noise_hr, image_gt_hr = random_cut(image_noise, image_gt, w=self.image_size)
        image_noise_lr = pixel_unshuffle(image_noise_hr,upscale_factor = self.upscale_factor)
        image_gt_lr = pixel_unshuffle(image_gt_hr,upscale_factor = self.upscale_factor)
        return image_noise_hr,image_noise_lr, image_gt_hr, image_gt_lr


    def __len__(self):
        return len(self.gt_path)