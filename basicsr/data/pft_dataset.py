import os
import glob
import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np

class PFTSRDataset(Dataset):
    def __init__(self, lq_dir, gt_dir, transform=None):
        self.lq_paths = sorted(glob.glob(os.path.join(lq_dir, '*.npy')))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
        assert len(self.lq_paths) == len(self.gt_paths), "Mismatch LQ/GT length"
        self.transform = transform

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq = np.load(self.lq_paths[idx]).astype(np.float32)  # (C, H, W)
        gt = np.load(self.gt_paths[idx]).astype(np.float32)

        lq = torch.from_numpy(lq)
        gt = torch.from_numpy(gt)

        if self.transform:
            lq = self.transform(lq)
            gt = self.transform(gt)

        return {"lq": lq, "gt": gt}