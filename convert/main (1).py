import os
import time
import torch
import imageio
import numpy as np

from convert.converter import torch2onnx, onnx2keras, keras2tflite
# from denoiser.networks.denoising_rgb import DenoiseNet
# from denoiser.utils import load_checkpoint
from model.KPN_DGF import KPN_DGF,Att_KPN_DGF,Att_Weight_KPN_DGF,Att_KPN_Wavelet_DGF
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint
from test_custom_DGF_mat import load_data

def convert_torch_to_onnx():
    img_path = '/home/dell/Downloads/FullTest/noisy/2_1.png'
    output_path = 'models/denoiser_rgb.onnx'

    input_node_names = ['input_image']
    output_nodel_names = ['output_image']

    # torch_model = DenoiseNet()
    # load_checkpoint(torch_model, model_path, 'cpu')
    checkpoint = load_checkpoint("../checkpoints/kpn_att_repeat_new/", False, 'latest')
    state_dict = checkpoint['state_dict']
    torch_model = Att_KPN_DGF(
        color=True,
        burst_length=4,
        blind_est=True,
        kernel_size=[5],
        sep_conv=False,
        channel_att=True,
        spatial_att=True,
        upMode="bilinear",
        core_bias=False
    )
    torch_model.load_state_dict(state_dict)

    img = imageio.imread(img_path)
    img = np.asarray(img, dtype=np.float32) / 255.

    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    image_noise, image_noise_hr = load_data(img_tensor, 4)
    # begin = time.time()
    # print(image_noise_batch.size())
    b, N, c, h, w = image_noise.size()
    feedData = image_noise.view(b, -1, h, w)
    print('Test forward pass')
    s = time.time()
    print("feedData  :",feedData.size())
    print("image_noise  ",image_noise[:, 0:4, ...].size())
    print("image_noise_hr  ",image_noise_hr.size())
    with torch.no_grad():
        _,enhanced_img_tensor = torch_model(feedData, image_noise[:, 0:4, ...],image_noise_hr)
        enhanced_img_tensor = torch.clamp(enhanced_img_tensor, 0, 1)
        enhanced_img = (enhanced_img_tensor.permute(0, 2, 3, 1).squeeze(0)
                        .cpu().detach().numpy())
        enhanced_img = np.clip(enhanced_img * 255, 0, 255).astype('uint8')
        imageio.imwrite('../img/denoised.jpg', enhanced_img)
    print('- Time: ', time.time() - s)

    print('Export to onnx format')
    s = time.time()
    # torch2onnx(torch_model, img_tensor, output_path, input_node_names,
    torch2onnx(torch_model, (feedData, image_noise[:, 0:4, ...],image_noise_hr), output_path, input_node_names,
               output_nodel_names, keep_initializers=False,
               verify_after_export=True)
    print('- Time: ', time.time() - s)


def convert_onnx_to_keras():
    onnx_model_path = 'models/denoiser_rgb.onnx'
    output_dir = 'models/keras_model'
    os.makedirs(output_dir, exist_ok=True)

    input_node_names = ['input_image']
    onnx2keras(onnx_model_path, input_node_names, output_dir)


def convert_keras_to_tflite():
    keras_model_dir = 'models/keras_model'
    output_path = 'models/denoiser_rgb.tflite'

    keras2tflite(keras_model_dir, output_path)


if __name__ == '__main__':
    convert_torch_to_onnx()
    convert_onnx_to_keras()
    convert_keras_to_tflite()
