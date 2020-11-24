import rawpy
import h5py
import numpy as np

if __name__ == "__main__":
    # image_path = "/home/dell/Downloads/0001_GT_RAW/0001_GT_RAW_003.MAT"
    # image_path = "/home/dell/Downloads/0001_NOISY_RAW/0001_NOISY_RAW_001.MAT"
    image_path = "/home/dell/Downloads/0001_METADATA_RAW/0001_METADATA_RAW_001.MAT"
    # raw_image = rawpy.imread(image_path).raw_image
    f = h5py.File(image_path)
    arrays_noise = {}
    print(f.items())
    for k, v in f.items():
        # print(k)
        arrays_noise[k] = np.array(v)
        # print(v.shape)
    noisy_img = arrays_noise['x']
    print(arrays_noise)
    print(np.max(noisy_img))
