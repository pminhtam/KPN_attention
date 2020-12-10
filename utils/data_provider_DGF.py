import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import glob
import torchvision.transforms as transforms
import math
from utils.data_transform import random_flip, random_rotate
##

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def random_cut(image_noise,image_gt,w,h=None):
    h = w if h is None else h
    nw = image_gt.size(-1) - w
    nh = image_gt.size(-2) - h
    if nw < 0 or nh < 0:
        raise RuntimeError("Image is to small {} for the desired size {}". \
                           format((image_gt.size(-1), image_gt.size(-2)), (w, h))
                           )

    idx_w = torch.randint(0, nw + 1, (1,))[0]
    idx_h = torch.randint(0, nh + 1, (1,))[0]
    image_noise_burst_crop = image_noise[:,idx_h:(idx_h+h), idx_w:(idx_w+w)]
    image_gt_crop = image_gt[:,idx_h:(idx_h+h), idx_w:(idx_w+w)]
    return image_noise_burst_crop,image_gt_crop

def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    # print(input.size())
    channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        channels, out_height, upscale_factor,
        out_width, upscale_factor)

    # channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(2, 4,0 , 1, 3).contiguous()
    return unshuffle_out.view(upscale_factor ** 2,channels, out_height, out_width)
class SingleLoader_DGF(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir,gt_dir,image_size=128,burst_length=16):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.noise_path = []
        self.burst_length = burst_length
        self.upscale_factor = int(math.sqrt(self.burst_length))
        for files_ext in IMG_EXTENSIONS:
            self.noise_path.extend(glob.glob(self.noise_dir +"/**/*" + files_ext,recursive=True))
        self.gt_path = []
        for files_ext in IMG_EXTENSIONS:
            self.gt_path.extend(glob.glob(self.gt_dir +"/**/*" + files_ext,recursive=True))
        
        if len(self.noise_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + self.noise_dir + "\n"
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
        image_gt_lr = pixel_unshuffle(image_gt_hr,upscale_factor = self.upscale_factor)
        return image_noise_hr,image_noise_lr, image_gt_hr, image_gt_lr


    def __len__(self):
        return len(self.noise_path)

class MultiLoader_DGF(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir, gt_dir, image_size=512,image_size_lr = 64):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.image_size_lr = image_size_lr
        self.noise_path = glob.glob(self.noise_dir + "/*")
        # for files_ext in IMG_EXTENSIONS:
        #     self.noise_path.extend(glob.glob(self.noise_dir + "/*" + files_ext))
        # self.gt_path = glob.glob(self.gt_dir + "/*")
        # for files_ext in IMG_EXTENSIONS:
        #     self.gt_path.extend(glob.glob(self.gt_dir + "/*" + files_ext))

        if len(self.noise_path) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.noise_dir + "\n"
                "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.resize_pytorch = transforms.Compose([transforms.ToPILImage(),transforms.Resize((self.image_size_lr,self.image_size_lr)),transforms.ToTensor()])

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

        path = self.noise_path[index]
        list_path = sorted(glob.glob(path+"/*"))[:8]


        name_folder_image = list_path[0].split("/")[-2].replace("NOISY_", "GT_")
        name_image = list_path[0].split("/")[-1].replace("NOISY_", "GT_")
        image_gt = random_flip(Image.open(os.path.join(self.gt_dir, name_folder_image,name_image)).convert('RGB'),rand_hflip,rand_vflip)
        image_gt = random_rotate(image_gt,rand_affine,angle)
        image_gt = self.transforms(image_gt)
        ############
        # Choose randcrop
        w = self.image_size
        h = self.image_size
        nw = image_gt.size(-1) - w
        nh = image_gt.size(-2) - h
        if nw < 0 or nh < 0:
            raise RuntimeError("Image is to small {} for the desired size {}". \
                               format((image_gt.size(-1), image_gt.size(-2)), (w, h))
                               )
        # print("nw  : ",nw)
        # idx_w = np.random.choice(nw + 1)
        idx_w = torch.randint(0,nw + 1,(1,))[0]
        # idx_h = np.random.choice(nh + 1)
        idx_h = torch.randint(0,nh + 1,(1,))[0]

        image_gt_hr_crop = image_gt[:,idx_h:(idx_h+h), idx_w:(idx_w+w)]

        image_noise_hr = [self.transforms(random_rotate(random_flip(Image.open(img_path).convert('RGB'),rand_hflip,rand_vflip),rand_affine,angle))[:,idx_h:(idx_h+h), idx_w:(idx_w+w)] for img_path in list_path]
        image_noise_lr = [self.resize_pytorch(image_noise_hr_i) for image_noise_hr_i in image_noise_hr]
        image_noise_hr_burst_crop = torch.stack(image_noise_hr, dim=0)
        image_noise_lr_burst_crop = torch.stack(image_noise_lr, dim=0)

        # image_noise_burst_crop, image_gt_crop = random_cut_burst(image_noise_burst,image_gt,w = self.image_size)
        return image_noise_hr_burst_crop,image_noise_lr_burst_crop, image_gt_hr_crop

    def __len__(self):
        return len(self.noise_path)

