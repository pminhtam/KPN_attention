import torch.utils.data as data
import torch
from PIL import Image
import os
from PIL import ImageFile
import glob
import torchvision.transforms as transforms
import math
from utils.data_transform import random_flip, random_rotate
from utils.data_provider_DGF import random_cut,pixel_unshuffle
from utils.create_noise import pipeline_configs, pipeline_param_ranges, _randomize_parameter, _create_pipeline
##
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma_plus = 0.1
        degrade_param = _randomize_parameter(pipeline_param_ranges)
        degrade_pipeline, self.target_pipeline = _create_pipeline(**{**pipeline_configs,
                                                                **degrade_param})
        self.degrade_pipeline = degrade_pipeline.to(device=self.device)
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
        if index >= len(self.gt_path):
            index = len(self.gt_path) - 1
        image_gt = Image.open(self.gt_path[index]).convert('RGB')
        # print(image_gt.size)
        nh = image_gt.size[-1] - self.image_size
        nw = image_gt.size[-2] - self.image_size
        while nw < 0 or nh < 0:
            print("del  : ",index,'  with size :', image_gt.size)
            del self.gt_path[index]
            if index >= len(self.gt_path):
                index = len(self.gt_path) -1
            image_gt = Image.open(self.gt_path[index]).convert('RGB')
            nh = image_gt.size[-1] - self.image_size
            nw = image_gt.size[-2] - self.image_size
        image_gt = random_flip(image_gt,rand_hflip,rand_vflip)
        image_gt = random_rotate(image_gt,rand_affine,angle)

        image_gt = self.transforms(image_gt)
        #### random cut
        idx_w = torch.randint(0, nw + 1, (1,))[0]
        idx_h = torch.randint(0, nh + 1, (1,))[0]
        image_gt_hr = image_gt[:,idx_h:(idx_h+self.image_size), idx_w:(idx_w+self.image_size)]
        type_rand = torch.rand(1)
        if type_rand < 0.5:
            noise = torch.randn(image_gt_hr.size())*self.sigma_plus*torch.rand(1)
            image_noise_hr = image_gt_hr + noise
        elif type_rand >= 0.5 and type_rand < 0.6:
            noise = torch.rand(image_gt_hr.size())*self.sigma_plus*torch.rand(1)
            image_noise_hr = image_gt_hr + noise
        else:
            # print("degrade_pipeline")
            noise = self.degrade_pipeline(image_gt_hr)[0]
            # print(noise.size())
            image_noise_hr = noise
        image_noise_lr = pixel_unshuffle(image_noise_hr,upscale_factor = self.upscale_factor)
        image_gt_lr = pixel_unshuffle(image_gt_hr,upscale_factor = self.upscale_factor)
        return image_noise_hr,image_noise_lr, image_gt_hr, image_gt_lr


    def __len__(self):
        return len(self.gt_path)