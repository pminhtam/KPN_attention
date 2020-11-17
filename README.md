## Attention Mechanism Enhanced Kernel Prediction Networks (AME-KPNs)
 Offical in https://github.com/z-bingo/Attention-Mechanism-Enhanced-KPN
 
 
The unofficial implementation of AME-KPNs in PyTorch, and our paper is accepted by ICASSP 2020 (oral), it is available at [http://arxiv.org/abs/1910.08313](http://arxiv.org/abs/1910.08313).

### News
- Support KPN (Kernel Prediction Networks), MKPN (Multi-Kernel Prediction Networks) by modifing the config file.
- The current version supports training on color images.
- Add Deep Guide Filter


### Requirements
- Python3
- PyTorch >= 1.0.0
- Scikit-image
- Numpy
- TensorboardX (needed tensorflow support)

## How to use it?
This repo. supports training on multiple GPUs and the default setting is also multi-GPU.  

Train

```
CUDA_VISIBLE_DEVICES=0,1 python train_eval_syn.py --noise_dir ../image/noise/ --gt_dir ../image/gt/ --image_size 512 --batch_size 1 --save_every 100 --loss_every 10 -nw 4 -c -m -ckpt att_kpn --model_type attKPN --restart```
```
If no `--restart`, the train process would be resumed.

Eval 

```
CUDA_VISIBLE_DEVICES=0,1 python test.py --noise_dir ../image/noise/ --gt_dir ../image/gt/ --image_size 512 -nw 4 -c -m -ckpt att_kpn --model_type attKPN```
```


### Citation
```
https://github.com/z-bingo/Attention-Mechanism-Enhanced-KPN
```

```
@article{zhang2019attention,
    title={Attention Mechanism Enhanced Kernel Prediction Networks for Denoising of Burst Images},
    author={Bin Zhang and Shenyao Jin and Yili Xia and Yongming Huang and Zixiang Xiong},
    year={2019},
    journal={arXiv preprint arXiv:1910.08313}
}
```
