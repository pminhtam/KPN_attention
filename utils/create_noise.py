import numpy as np
from utils.noise_generation.pipeline import ImageDegradationPipeline
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
def _create_pipeline(exp_adjustment, poisson_k, read_noise_sigma, chromatic_aberration, motion_blur_dir, jpeg_quality,
                     denoise_sigma_s, denoise_sigma_r, denoise_color_sigma_ratio, unsharp_amount, denoise_color_only,
                     demosaick, denoise, jpeg_compression, use_motion_blur, use_chromatic_aberration, use_unsharp_mask,
                     exposure_correction, quantize, quantize_bits=8, denoise_guide_transform=None, denoise_n_iter=1,
                     demosaick_use_median=False, demosaick_n_iter=0, use_median_denoise=False, median_before_bilateral=False,
                     denoise_median=None, denoise_median_ratio=1.0, denoise_median_n_iter=1, demosaicked_input=True,
                     log_blackpts=0.004, bilateral_class="DenoisingSKImageBilateralNonDifferentiable",
                     demosaick_class="AHDDemosaickingNonDifferentiable", demosaick_ahd_delta=2.0, demosaick_ahd_sobel_sz=3,
                     demosaick_ahd_avg_sz=3, use_wavelet=False, wavelet_family="db2", wavelet_sigma=None, wavelet_th_method="BayesShrink",
                     wavelet_levels=None, motion_blur=None, motionblur_th=None, motionblur_boost=None, motionblur_segment=1,
                     debug=False, bayer_crop_phase=None, saturation=None, use_autolevel=False, autolevel_max=1.5, autolevel_blk=1,
                     autolevel_wht=99, denoise_color_range_ratio=1, wavelet_last=False, wavelet_threshold=None, wavelet_filter_chrom=True,
                     post_tonemap_class=None, post_tonemap_amount=None,  pre_tonemap_class=None, pre_tonemap_amount=None,
                     post_tonemap_class2=None, post_tonemap_amount2=None, repair_hotdead_pixel=False, hot_px_th=0.2,
                     white_balance=False, white_balance_temp=6504, white_balance_tint=0, use_tone_curve3zones=False,
                     tone_curve_highlight=0.0, tone_curve_midtone=0.0, tone_curve_shadow=0.0, tone_curve_midshadow=None,
                     tone_curve_midhighlight=None, unsharp_radius=4.0, unsharp_threshold=3.0, **kwargs):
    # Define image degradation pipeline
    # add motion blur and chromatic aberration
    configs_degrade = []
    # Random threshold
    if demosaicked_input:
        # These are features that only make sense to simulate in
        # demosaicked input.
        if use_motion_blur:
            configs_degrade += [
                ('MotionBlur', {'amt': motion_blur,
                                'direction': motion_blur_dir,
                                'kernel_sz': None,
                                'dynrange_th': motionblur_th,
                                'dynrange_boost': motionblur_boost,
                                }
                 )
            ]
        if use_chromatic_aberration:
            configs_degrade += [
                ('ChromaticAberration', {'scaling': chromatic_aberration}),
            ]

    configs_degrade.append(('ExposureAdjustment', {'nstops': exp_adjustment}))
    if demosaicked_input:
        if demosaick:
            configs_degrade += [
                ('BayerMosaicking', {}),
            ]
            mosaick_pattern = 'bayer'
        else:
            mosaick_pattern = None
    else:
        mosaick_pattern = 'bayer'

    # Add artificial noise.
    configs_degrade += [
        ('PoissonNoise', {'sigma': poisson_k, 'mosaick_pattern': mosaick_pattern}),
        ('GaussianNoise', {'sigma': read_noise_sigma, 'mosaick_pattern': mosaick_pattern}),
    ]

    if quantize:
        configs_degrade += [
            ('PixelClip', {}),
            ('Quantize', {'nbits': quantize_bits}),
        ]
    if repair_hotdead_pixel:
        configs_degrade += [
            ("RepairHotDeadPixel", {"threshold": hot_px_th}),
        ]

    if demosaick:
        configs_degrade += [
            (demosaick_class, {'use_median_filter': demosaick_use_median,
                               'n_iter': demosaick_n_iter,
                               'delta': demosaick_ahd_delta,
                               'sobel_sz': demosaick_ahd_sobel_sz,
                               'avg_sz': demosaick_ahd_avg_sz,
                               }),
            ('PixelClip', {}),
        ]
    if white_balance:
        configs_degrade += [
            ('WhiteBalanceTemperature', {"new_temp": white_balance_temp,
                                         "new_tint": white_balance_tint,
                                         }),
        ]
    if pre_tonemap_class is not None:
        kw = "gamma" if "Gamma" in pre_tonemap_class else "amount"
        configs_degrade += [
            (pre_tonemap_class, {kw: pre_tonemap_amount})
        ]
    if use_autolevel:
        configs_degrade.append(('AutoLevelNonDifferentiable', {'max_mult': autolevel_max,
                                                               'blkpt': autolevel_blk,
                                                               'whtpt': autolevel_wht,
                                                               }))
    denoise_list = []
    if denoise:
        denoise_list.append([
            ('PixelClip', {}),
            (bilateral_class, {'sigma_s': denoise_sigma_s,
                               'sigma_r': denoise_sigma_r,
                               'color_sigma_ratio': denoise_color_sigma_ratio,
                               'color_range_ratio': denoise_color_range_ratio,
                               'filter_lum': not denoise_color_only,
                               'n_iter': denoise_n_iter,
                               'guide_transform': denoise_guide_transform,
                               '_bp': log_blackpts,
                               }),
            ('PixelClip', {}),
        ])

    if use_median_denoise:
        # TODO: Fix this.
        # Special value because our config can't specify list of list
        if denoise_median == -1:
            denoise_median = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        if debug:
            print("Denoising with Median Filter")
        denoise_list.append([
            ('DenoisingMedianNonDifferentiable', {'neighbor_sz': denoise_median,
                                                  'color_sigma_ratio': denoise_median_ratio,
                                                  'n_iter': denoise_median_n_iter,
                                                  }),
        ])

    if median_before_bilateral:
        denoise_list = denoise_list[::-1]
    if use_wavelet:
        # always do wavelet first.
        wavelet_config = [
            ('PixelClip', {}),
            ("DenoisingWaveletNonDifferentiable", {'sigma_s': wavelet_th_method,
                                                   'sigma_r': wavelet_sigma,
                                                   'color_sigma_ratio': wavelet_family,
                                                   'filter_lum': True,
                                                   'n_iter': wavelet_levels,
                                                   'guide_transform': denoise_guide_transform,
                                                   '_bp': wavelet_threshold,
                                                   'filter_chrom': wavelet_filter_chrom,
                                                   }),
            ('PixelClip', {}),
        ]
        if wavelet_last:
            denoise_list.append(wavelet_config)
        else:
            denoise_list.insert(0, wavelet_config)

    for i in range(len(denoise_list)):
        configs_degrade += denoise_list[i]
    if post_tonemap_class is not None:
        kw = "gamma" if "Gamma" in post_tonemap_class else "amount"
        configs_degrade += [
            (post_tonemap_class, {kw: post_tonemap_amount})
        ]
    if post_tonemap_class2 is not None:
        kw = "gamma" if "Gamma" in post_tonemap_class2 else "amount"
        configs_degrade += [
            (post_tonemap_class2, {kw: post_tonemap_amount2})
        ]
    if use_tone_curve3zones:
        ctrl_val = [t for t in [tone_curve_shadow,
                                tone_curve_midshadow,
                                tone_curve_midtone,
                                tone_curve_midhighlight,
                                tone_curve_highlight] if t is not None]
        configs_degrade += [
            ('ToneCurveNZones', {'ctrl_val': ctrl_val,
                                 }),
            ('PixelClip', {}),
        ]

    if use_unsharp_mask:
        configs_degrade += [
            ('Unsharpen', {'amount': unsharp_amount,
                           'radius': unsharp_radius,
                           'threshold': unsharp_threshold}),
            ('PixelClip', {}),
        ]

    if saturation is not None:
        configs_degrade.append(('Saturation', {'value': saturation}))

    # things that happens after camera apply denoising, etc.
    if jpeg_compression:
        configs_degrade += [
            ('sRGBGamma', {}),
            ('Quantize', {'nbits': 8}),
            ('PixelClip', {}),
            ('JPEGCompression', {"quality": jpeg_quality}),
            ('PixelClip', {}),
            ('UndosRGBGamma', {}),
            ('PixelClip', {}),
        ]
    else:
        if quantize:
            configs_degrade += [
                ('Quantize', {'nbits': 8}),
                ('PixelClip', {}),
            ]

    if exposure_correction:
        # Finally do exposure correction of weird jpeg-compressed image to get crappy images.
        configs_degrade.append(('ExposureAdjustment', {'nstops': -exp_adjustment}))
        target_pipeline = None
    else:
        configs_target = [
            ('ExposureAdjustment', {'nstops': exp_adjustment}),
            ('PixelClip', {}),
        ]
        target_pipeline = ImageDegradationPipeline(configs_target)

    configs_degrade.append(('PixelClip', {}))
    if debug:
        print('Final config:')
        print('\n'.join([str(c) for c in configs_degrade]))

    degrade_pipeline = ImageDegradationPipeline(configs_degrade)
    return degrade_pipeline, target_pipeline

def _random_log_uniform(legacy_uniform,a, b):
    if legacy_uniform:
        return np.random.uniform(a, b)
    val = np.random.uniform(np.log(a), np.log(b))
    return np.exp(val)

def _randomize_parameter(pipeline_param_ranges):
        if "use_log_uniform" in pipeline_configs:
             legacy_uniform = not pipeline_configs["use_log_uniform"]
        else:
             legacy_uniform = True

        exp_adjustment = np.random.uniform( pipeline_param_ranges["min_exposure_adjustment"],
                                            pipeline_param_ranges["max_exposure_adjustment"])
        poisson_k =  _random_log_uniform(legacy_uniform,pipeline_param_ranges["min_poisson_noise"],
                                              pipeline_param_ranges["max_poisson_noise"])
        read_noise_sigma =  _random_log_uniform(legacy_uniform, pipeline_param_ranges["min_gaussian_noise"],
                                                     pipeline_param_ranges["max_gaussian_noise"])
        chromatic_aberration = np.random.uniform( pipeline_param_ranges["min_chromatic_aberration"],
                                                  pipeline_param_ranges["max_chromatic_aberration"])
        motionblur_segment = np.random.randint( pipeline_param_ranges["min_motionblur_segment"],
                                                pipeline_param_ranges["max_motionblur_segment"])
        motion_blur = []
        motion_blur_dir = []
        for i in range(motionblur_segment):
            motion_blur.append(np.random.uniform( pipeline_param_ranges["min_motion_blur"],
                                                  pipeline_param_ranges["max_motion_blur"])
                               )
            motion_blur_dir.append(np.random.uniform(0.0, 360.0))
        jpeg_quality = np.random.randint( pipeline_param_ranges["min_jpeg_quality"],
                                          pipeline_param_ranges["max_jpeg_quality"])
        denoise_sigma_s =  _random_log_uniform(legacy_uniform,pipeline_param_ranges["min_denoise_sigma_s"],
                                                    pipeline_param_ranges["max_denoise_sigma_s"])
        denoise_sigma_r =  _random_log_uniform(legacy_uniform,pipeline_param_ranges["min_denoise_sigma_r"],
                                                    pipeline_param_ranges["max_denoise_sigma_r"])
        denoise_color_sigma_ratio =  _random_log_uniform(legacy_uniform,
             pipeline_param_ranges["min_denoise_color_sigma_ratio"],
             pipeline_param_ranges["max_denoise_color_sigma_ratio"])
        denoise_color_range_ratio =  _random_log_uniform(legacy_uniform,
             pipeline_param_ranges["min_denoise_color_range_ratio"],
             pipeline_param_ranges["max_denoise_color_range_ratio"])
        unsharp_amount = np.random.uniform( pipeline_param_ranges["min_unsharp_amount"],
                                            pipeline_param_ranges["max_unsharp_amount"])
        denoise_median_sz = np.random.randint( pipeline_param_ranges["min_denoise_median_sz"],
                                               pipeline_param_ranges["max_denoise_median_sz"])
        quantize_bits = np.random.randint( pipeline_param_ranges["min_quantize_bits"],
                                           pipeline_param_ranges["max_quantize_bits"])
        wavelet_sigma = np.random.uniform( pipeline_param_ranges["min_wavelet_sigma"],
                                           pipeline_param_ranges["max_wavelet_sigma"])
        motionblur_th = np.random.uniform( pipeline_param_ranges["min_motionblur_th"],
                                           pipeline_param_ranges["max_motionblur_th"])
        motionblur_boost =  _random_log_uniform( legacy_uniform,pipeline_param_ranges["min_motionblur_boost"],
                                                     pipeline_param_ranges["max_motionblur_boost"])
        return dict(
            exp_adjustment=exp_adjustment,
            poisson_k=poisson_k,
            read_noise_sigma=read_noise_sigma,
            chromatic_aberration=chromatic_aberration,
            motion_blur=motion_blur,
            motion_blur_dir=motion_blur_dir,
            jpeg_quality=jpeg_quality,
            denoise_sigma_s=denoise_sigma_s,
            denoise_sigma_r=denoise_sigma_r,
            denoise_color_sigma_ratio=denoise_color_sigma_ratio,
            denoise_color_range_ratio=denoise_color_range_ratio,
            unsharp_amount=unsharp_amount,
            denoise_median=denoise_median_sz,
            quantize_bits=quantize_bits,
            wavelet_sigma=wavelet_sigma,
            motionblur_th=motionblur_th,
            motionblur_boost=motionblur_boost,
        )

pipeline_configs = {'denoise': True, 'demosaick': True, 'jpeg_compression': True, 'use_unsharp_mask': True,
                    'use_motion_blur': False, 'use_chromatic_aberration': False, 'denoise_color_only': False,
                    'exposure_correction': False, 'quantize': True, 'denoise_guide_transform': 0.5,
                    'use_median_denoise': True, 'use_wavelet': False, 'use_log_uniform': True,
                    'median_before_bilateral': True, 'denoise_n_iter': 1, 'demosaick_use_median': False,
                    'demosaick_n_iter': 0, 'wavelet_family': 'db2', 'wavelet_th_method': 'BayesShrink',
                    'wavelet_levels': None, 'bayer_crop_phase': None, 'repair_hotdead_pixel': False,
                    'hot_px_th': 0.2}
pipeline_param_ranges = {'min_gaussian_noise': 0.002, 'max_gaussian_noise': 0.1, 'min_poisson_noise': 0.02,
                         'max_poisson_noise': 0.2, 'min_jpeg_quality': 4, 'max_jpeg_quality': 8,
                         'min_denoise_sigma_s': 0.25, 'max_denoise_sigma_s': 1.0, 'min_denoise_sigma_r': 0.1,
                         'max_denoise_sigma_r': 1.0, 'min_denoise_color_sigma_ratio': 4.0,
                         'max_denoise_color_sigma_ratio': 32.0, 'min_denoise_color_range_ratio': 0.1,
                         'max_denoise_color_range_ratio': 0.5, 'min_unsharp_amount': 0.0, 'max_unsharp_amount': 0.25,
                         'min_denoise_median_sz': -1, 'max_denoise_median_sz': 0, 'min_exposure_adjustment': 0.0,
                         'max_exposure_adjustment': 0.0, 'min_motion_blur': 0.0, 'max_motion_blur': 0.0,
                         'min_chromatic_aberration': 1.0, 'max_chromatic_aberration': 1.0, 'min_quantize_bits': 14,
                         'max_quantize_bits': 15, 'min_wavelet_sigma': 0.003, 'max_wavelet_sigma': 0.015,
                         'min_motionblur_th': 0.99, 'max_motionblur_th': 0.997, 'min_motionblur_boost': 5.0,
                         'max_motionblur_boost': 100.0, 'min_motionblur_segment': 1, 'max_motionblur_segment': 4}
if __name__ =="__main__":
    degrade_param = _randomize_parameter(pipeline_param_ranges)
    degrade_pipeline, target_pipeline = _create_pipeline(**{**pipeline_configs,
                                                                 **degrade_param})
    print(degrade_pipeline)
    print(target_pipeline)
    img = (transforms.ToTensor()(Image.open('/home/dell/Downloads/gt/0001_GT_SRGB/0001_GT_SRGB_002.PNG')))
    print(img.size())
    img = img[:,:2048,:2048].unsqueeze(0)
    print(img.size())

    img_noise = degrade_pipeline(img)
    trans = transforms.ToPILImage()
    # plt.figure(figsize=(30, 9))
    print(img.size())
    print(img_noise.size())
    plt.subplot(1,2,1)
    plt.imshow(np.array(trans(img[0])))
    plt.subplot(1,2,2)
    plt.imshow(np.array(trans(img_noise[0])))
    plt.show()