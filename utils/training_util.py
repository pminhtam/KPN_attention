import numpy as np
import glob
import torch
import shutil
import os
import cv2
import numbers
import skimage
from collections import OrderedDict
from skimage.measure import compare_psnr,compare_ssim


class MovingAverage(object):
    def __init__(self, n):
        self.n = n
        self._cache = []
        self.mean = 0

    def update(self, val):
        self._cache.append(val)
        if len(self._cache) > self.n:
            del self._cache[0]
        self.mean = sum(self._cache) / len(self._cache)

    def get_value(self):
        return self.mean


def save_checkpoint(state, is_best, checkpoint_dir, n_iter, max_keep=10):
    filename = os.path.join(checkpoint_dir, "{:06d}.pth.tar".format(n_iter))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        os.path.join(checkpoint_dir,
                                     'model_best.pth.tar'))
    files = sorted(os.listdir(checkpoint_dir))
    rm_files = files[0:max(0, len(files) - max_keep)]
    for f in rm_files:
        os.remove(os.path.join(checkpoint_dir, f))

def _represent_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def load_checkpoint(checkpoint_dir,cuda = False, best_or_latest='best'):
    if best_or_latest == 'best':
        checkpoint_file = os.path.join(checkpoint_dir, 'model_best.pth.tar')
    elif isinstance(best_or_latest, numbers.Number):
        checkpoint_file = os.path.join(checkpoint_dir,
                                       '{:06d}.pth.tar'.format(best_or_latest))
        if not os.path.exists(checkpoint_file):
            files = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
            basenames = [os.path.basename(f).split('.')[0] for f in files]
            iters = sorted([int(b) for b in basenames if _represent_int(b)])
            raise ValueError('Available iterations are ({} requested): {}'.format(best_or_latest, iters))
    else:
        files = glob.glob(os.path.join(checkpoint_dir, '*.pth.tar'))
        basenames = [os.path.basename(f).split('.')[0] for f in files]
        iters = sorted([int(b) for b in basenames if _represent_int(b)])
        checkpoint_file = os.path.join(checkpoint_dir,
                                       '{:06d}.pth.tar'.format(iters[-1]))
    if cuda:
        load_result = torch.load(checkpoint_file)
    else:
        load_result = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    return load_result


def load_statedict_runtime(checkpoint_dir, best_or_latest='best'):
    # This function grabs state_dict from checkpoint, and do modification
    # to the weight name so that it can be load at runtime.
    # During training nn.DataParallel adds 'module.' to the name,
    # which doesn't exist at test time.
    ckpt = load_checkpoint(checkpoint_dir, best_or_latest)
    state_dict = ckpt['state_dict']
    global_iter = ckpt['global_iter']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove `module.`
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict, global_iter


def prep_and_vis_flow(flow, flow_visualizer, max_flow=None):
    flow = flow_visualizer(flow[0, :, :, :], max_flow=max_flow)
    flow = flow.cpu().data.numpy()
    return flow


def put_text_on_img(image, text, loc=(20, 100), color=(1, 0, 0)):
    """ Put text on flow

    Args:
        image: numpy array of dimension (3, h, w)
        text: text to put on.
        loc: ibottom-left location of text in (x, y) from top-left of image.
        color: color of the text.
    Returns:
        image with text written on it.
    """
    image = np.array(np.moveaxis(image, 0, -1)).copy()
    cv2.putText(image, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    return np.moveaxis(image, -1, 0)

def torch2numpy(tensor, gamma=None):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to 0 - 255
    if gamma is not None:
        tensor = torch.pow(tensor, gamma)
    tensor *= 255.0
    while len(tensor.size()) < 4:
        tensor = tensor.unsqueeze(1)
    return tensor.permute(0, 2, 3, 1).cpu().data.numpy()


def calculate_psnr(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    psnr = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        psnr += compare_psnr(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...],
                                             data_range=255)
        n += 1.0
    return psnr / n


def calculate_ssim(output_img, target_img):
    target_tf = torch2numpy(target_img)
    output_tf = torch2numpy(output_img)
    ssim = 0.0
    n = 0.0
    for im_idx in range(output_tf.shape[0]):
        ssim += compare_ssim(target_tf[im_idx, ...],
                                             output_tf[im_idx, ...],
                                             multichannel=True,
                                             data_range=255)
        n += 1.0
    return ssim / n
