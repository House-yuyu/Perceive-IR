import os.path
import random

import numpy as np
import torch
from PIL import Image
from basicsr.data.transforms import paired_random_crop, augment, mod_crop
from torchvision.transforms.functional import normalize
from basicsr.utils import scandir, img2tensor, FileClient, imfrombytes
from basicsr.utils.matlab_functions import bgr2ycbcr
from torch.utils import data
import logging
import pandas as pd

"""
CSV file structure:{
            'gt_path':
            'lq_path':
            'caption':	
            'degradation':
                    }
"""
class CsvDatasetV2(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        ** root_path : 项目根目录
        ** csv_file_path : LQ path | caption | degradation 相对路径，相对项目根目录
        ** require_text : True or False

        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.

        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """
    def __init__(self, opt):
        super(CsvDatasetV2, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        ######
        self.require_text = opt['require_text']
        self.root_path = opt['root_path'] if opt['root_path'] is not None else ''
        csv_file_path = opt['csv_file_path']
        logging.debug(f'Loading csv data from {csv_file_path}.')

        csv_file_path = os.path.join(self.root_path, csv_file_path)
        df = pd.read_csv(csv_file_path, sep="\t")
        df["gt_path"] = df["gt_path"].fillna("")     

        self.images_LQ = df['lq_path'].tolist()                # datasets/universal/train/rainy/LQ/norain-1000x2.png
        self.images_GT = df['gt_path'].tolist()                # datasets/universal/train/rainy/LQ/norain-1000x2.png
        self.captions = df['caption'].tolist()
        self.degradation = df['degradation'].tolist()

        self.deg_map = {
            'hazy': 'haze',
            'rainy': 'rain',
            'raindrop': 'raindrop',
            'snowy': 'snow',
            'snow': 'snow',
            'rain and haze': 'rain and haze',
            'Rain and Fog': 'rain and fog',
            'noisy': 'noise',
            'Rain': 'rain'
        }

    def __len__(self):
        return len(self.captions)

    def __get_img__(self, gt_path, lq_path):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # print('-----------------')
        # print(self.root_path)
        # print(gt_path)

        gt_path = os.path.join(self.root_path, gt_path)
        lq_path = os.path.join(self.root_path, lq_path)

        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        # if self.opt['phase'] != 'train':
        #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        ## mod crop
        if self.opt['phase'] != 'train':
                img_gt = mod_crop(img_gt, self.opt['mod_crop_scale'])
                img_lq = mod_crop(img_lq, self.opt['mod_crop_scale'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return img_lq, img_gt

    def __getitem__(self, idx):
        images_LQ, images_GT = self.__get_img__(gt_path=self.images_GT[idx], lq_path=self.images_LQ[idx])
        caption = str(self.captions[idx])
        degradation = str(self.degradation[idx]).rstrip()        

        if not self.require_text:
            caption = ''
        return {
            'lq': images_LQ,
            'gt': images_GT,
            'text': caption,
            'img_path': self.images_GT[idx],
            'degradation': degradation,
            'text_path': ""
        }

