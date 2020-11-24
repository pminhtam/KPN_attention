import torch.utils.data as data
import torch
import os
import os.path
import glob
import torchvision.transforms as transforms
import numpy as np
from utils.data_provider import random_cut
from raw_process.raw_util import read_raw
##

IMG_EXTENSIONS = [
    '.MAT'
]

class MultiLoader(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir, gt_dir, image_size=512):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
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

        # self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue
        """

        path = self.noise_path[index]
        list_path = sorted(glob.glob(path + "/*MAT"))[:8]

        name_folder_image = list_path[0].split("/")[-2].replace("NOISY_", "GT_")
        name_image = list_path[0].split("/")[-1].replace("NOISY_", "GT_")
        image_gt = read_raw(os.path.join(self.gt_dir, name_folder_image, name_image))
        # print("gt shape   : ",image_gt.shape)
        image_gt = torch.from_numpy(image_gt)
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
        idx_w = torch.randint(0, nw + 1, (1,))[0]
        # idx_h = np.random.choice(nh + 1)
        idx_h = torch.randint(0, nh + 1, (1,))[0]

        # idx_w = torch.randint(nw + 1)
        # idx_h = torch.randint(nh+1)
        # print(idx_w)
        # print(idx_h)
        ##########
        # print("image_gt shape   : ",image_gt.shape)

        image_gt_crop = image_gt[idx_h:(idx_h + h), idx_w:(idx_w + w)]
        # print("image_gt_crop shape   : ",image_gt_crop.shape)

        image_noise = [torch.from_numpy(read_raw(img_path))[idx_h:(idx_h + h), idx_w:(idx_w + w)] for
                       img_path in list_path]
        image_noise_burst_crop = torch.stack(image_noise, dim=0)

        # image_noise_burst_crop, image_gt_crop = random_cut_burst(image_noise_burst,image_gt,w = self.image_size)
        return image_noise_burst_crop, image_gt_crop

    def __len__(self):
        return len(self.noise_path)

