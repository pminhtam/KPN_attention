import cv2
import numpy as np
import scipy.io as sio


def read_metadata(metadata_file_path):
    # metadata
    meta = sio.loadmat(metadata_file_path)
    meta = meta['metadata'][0, 0]
    # black_level = float(meta['black_level'][0, 0])
    # white_level = float(meta['white_level'][0, 0])
    bayer_pattern = get_bayer_pattern(meta)  # meta['bayer_pattern'].tolist()
    bayer_2by2 = (np.asarray(bayer_pattern) + 1).reshape((2, 2)).tolist()
    # nlf = meta['nlf']
    # shot_noise = nlf[0, 2]
    # read_noise = nlf[0, 3]
    wb = get_wb(meta)
    # cst1 = meta['cst1']
    cst1, cst2 = get_csts(meta)
    # cst2 = cst2.reshape([3, 3])  # use cst2 for rendering, TODO: interpolate between cst1 and cst2
    iso = get_iso(meta)
    cam = get_cam(meta)
    return meta, bayer_2by2, wb, cst2, iso, cam


def get_iso(metadata):
    try:
        iso = metadata['ISOSpeedRatings'][0][0]
    except:
        try:
            iso = metadata['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        except:
            raise Exception('ISO not found.')
    return iso


def get_cam(metadata):
    model = metadata['Make'][0]
    # cam_ids = [0, 1, 3, 3, 4]  # IP, GP, S6, N6, G4
    cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
    return cam_dict[model]


def get_bayer_pattern(metadata):
    bayer_id = 33422
    bayer_tag_idx = 1
    try:
        unknown_tags = metadata['UnknownTags']
        if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
            bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
        else:
            raise Exception
    except:
        try:
            unknown_tags = metadata['SubIFDs'][0, 0]['UnknownTags'][0, 0]
            if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
                bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
            else:
                raise Exception
        except:
            try:
                unknown_tags = metadata['SubIFDs'][0, 1]['UnknownTags']
                if unknown_tags[1]['ID'][0][0][0] == bayer_id:
                    bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
                else:
                    raise Exception
            except:
                print('Bayer pattern not found. Assuming RGGB.')
                bayer_pattern = [1, 2, 2, 3]
    return bayer_pattern


def get_wb(metadata):
    return metadata['AsShotNeutral']


def get_csts(metadata):
    return metadata['ColorMatrix1'].reshape((3, 3)), metadata['ColorMatrix2'].reshape((3, 3))

from glob import glob
if __name__ == "__main__":
    files_ = glob('/home/dell/Downloads/0001_METADATA_RAW/*')
    for fi in files_:
        re, bayer_2by2, wb, cst2, iso, cam = read_metadata(fi)
        print(re['Make'])
        # print(wb)
        exit(0)