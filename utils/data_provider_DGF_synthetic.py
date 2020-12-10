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
class SingleLoader_DGF(data.Dataset):
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

        image_noise = random_flip(Image.open(self.noise_path[index]).convert('RGB'),rand_hflip,rand_vflip)
        image_noise = random_rotate(image_noise,rand_affine,angle)

        # name_image_gt = self.noise_path[index].split("/")[-1]
        # image_folder_name_gt = self.noise_path[index].split("/")[-2].replace("NOISY_","GT_")
        # image_gt = random_flip(Image.open(os.path.join(self.gt_dir, name_image_gt)).convert('RGB'), rand_hflip, rand_vflip)
        name_image_gt = self.noise_path[index].split("/")[-1].replace("NOISY_","GT_")
        image_folder_name_gt = self.noise_path[index].split("/")[-2].replace("NOISY_","GT_")
        image_gt = random_flip(Image.open(os.path.join(self.gt_dir,image_folder_name_gt, name_image_gt)).convert('RGB'),rand_hflip,rand_vflip)
        image_gt = random_rotate(image_gt,rand_affine,angle)

        image_noise = self.transforms(image_noise)
        image_gt = self.transforms(image_gt)
        image_noise_hr, image_gt_hr = random_cut(image_noise, image_gt, w=self.image_size)
        image_noise_lr = pixel_unshuffle(image_noise_hr,upscale_factor = self.upscale_factor)
        return image_noise_hr,image_noise_lr, image_gt_hr,


    def __len__(self):
        return len(self.noise_path)