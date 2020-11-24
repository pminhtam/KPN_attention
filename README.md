## Attention Mechanism Enhanced Kernel Prediction Networks (AME-KPNs)
 Offical in [Attention-Mechanism-Enhanced-KPN](https://github.com/z-bingo/Attention-Mechanism-Enhanced-KPN)
 
 
The unofficial implementation of AME-KPNs in PyTorch, and paper is accepted by ICASSP 2020 (oral), it is available at [http://arxiv.org/abs/1910.08313](http://arxiv.org/abs/1910.08313).

## Data

Use [SIDD dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) to train. 

Have two folder : noisy image and ground true image

Input folder have struct :

```
/
    /noise
        /[scene_instance]
            /[image].PNG
    /gt
        /[scene_instance]
            /[image].PNG
```

## Train
This repo. supports training on multiple GPUs.  

Train

```
CUDA_VISIBLE_DEVICES=0,1 python train_eval_syn.py --noise_dir ../image/noise/ --gt_dir ../image/gt/ --image_size 512 --batch_size 1 --save_every 100 --loss_every 10 -nw 4 -c -m -ckpt att_kpn --model_type attKPN --restart```
```
If no `--restart`, the train process would be resumed.


Train Deep Guide Filter

```
CUDA_VISIBLE_DEVICES=0,1 python train_eval_syn_DGF.py --noise_dir ../image/noise/ --gt_dir ../image/gt/ --image_size 512 --batch_size 1 --burst_length 16 --save_every 100 --loss_every 10 -nw 4 -c -m -ckpt att_kpn --model_type attKPN --restart```

```

## Eval

Eval 

```
CUDA_VISIBLE_DEVICES=0,1 python test.py --noise_dir ../image/noise/ --gt_dir ../image/gt/ --image_size 512 -nw 4 -c -m -ckpt att_kpn --model_type attKPN```
```

Eval with custome data : data have two folder image : *noise* and *gt*. 

Image will save in folder *-s* after predicted.

```
CUDA_VISIBLE_DEVICES=1 python test_custom_DGF.py -n ../FullTest/noisy/ -g ../FullTest/clean/ -b 4 -c -ckpt att_kpn_dgf_4_new -m attKPN -s img/att_kpn_dgf_4_new
```

### News
- Support KPN (Kernel Prediction Networks), MKPN (Multi-Kernel Prediction Networks)
- The current version supports training on color images.
- Add Deep Guide Filter
- Add noise estimate model to end-to-end denoising model

## Name 

*_custom  : load image from unstruct folder, print or save image for report

*_split : load one image and split image into burst image. 

*_DGF : model with Deep Guide Filter

*_noise : model with noise estimate 

### Requirements
```
pip install -r requirments.txt
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
