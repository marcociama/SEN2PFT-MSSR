import os
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset per image restoration, compatibile anche con .npy.

    Argomenti:
        opt (dict): Configurazione del dataset. Deve contenere:
            - dataroot_gt (str): cartella delle ground-truth
            - dataroot_lq (str): cartella delle low-quality
            - io_backend (dict): backend (e.g. {'type': 'disk'})
            - filename_tmpl (str): template per i nomi, default '{}'
            - gt_size (int): dimensione del patch crop per GT
            - use_hflip (bool), use_rot (bool), scale (int), phase ('train'/'val'), ecc.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file_client solo per immagini non .npy
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # costruzione coppie lq, gt
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = [p for p in paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
                if p['lq_path'].endswith('.npy') and p['gt_path'].endswith('.npy')]

    def __getitem__(self, index):
        scale = self.opt['scale']
        phase = self.opt['phase']

        # Prendo i path
        lq_path = self.paths[index]['lq_path']
        gt_path = self.paths[index]['gt_path']

        # carica GT
        img_gt = np.load(gt_path,allow_pickle=True).astype(np.float32)
        if img_gt.ndim == 3 and img_gt.shape[0] <= 10:
            img_gt = np.transpose(img_gt, (1, 2, 0))

        # carica LQ
        img_lq = np.load(lq_path,allow_pickle=True).astype(np.float32)
        if img_lq.ndim == 3 and img_lq.shape[0] <= 10:
            img_lq = np.transpose(img_lq, (1, 2, 0))

        # train augmentation
        if phase == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        if self.opt.get('color', '') == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        if phase != 'train':
            # Taglio GT affinché sia esattamente scale * dimensione LQ
            h_lq, w_lq = img_lq.shape[0], img_lq.shape[1]
            img_gt = img_gt[0: h_lq * scale, 0: w_lq * scale, :]

        # converto in tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        # sottrazione della media dataset-wise
        if self.mean is not None:
            mean_tensor = torch.tensor(self.mean, dtype=img_gt.dtype, device=img_gt.device)
            mean_tensor = mean_tensor.view(-1, 1, 1)  # reshape in (C,1,1) per broadcast
            img_gt = img_gt - mean_tensor
            img_lq = img_lq - mean_tensor

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)