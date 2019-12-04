import os
import matplotlib as plt
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import nibabel as nib

import _pickle as pickle


class SyntheticData(data.Dataset):
    """ Class defined to handle the synthetic dataset
        derived from pytorch's Dataset class"""

    # This will depend on the image_paths_file that
    # I should input
    def __init__(self, root_path):
        self.root_dir_name = os.path.dirname(root_path)
        self.raw_dir_name = os.path.join(self.root_dir_name, "raw/")
        self.seg_dir_name = os.path.join(self.root_dir_name, "seg/")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            return [self[i] for i in range(idx)]
        else:
            raw_img_name = os.path.join(self.raw_dir_name, ("%d.nii.gz" % idx))
            seg_img_name = os.path.join(self.seg_dir_name, ("%d.nii.gz" % idx))
            
            raw_img = nib.load(raw_img_name).get_data()
            print(nib.load(raw_img_name).header)
            seg_img = nib.load(seg_img_name).get_data()
            
            sample = {"image": raw_img, "segmentation": seg_img}
            
            return sample
        
    def __len__(self):
        path, dirs, files = next(os.walk(self.raw_dir_name))
        return len(onlyfiles)



