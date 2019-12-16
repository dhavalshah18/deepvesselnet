import os
import matplotlib as plt
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import nibabel as nib
import random

import _pickle as pickle


class SyntheticData(data.Dataset):
    """ 
    Class defined to handle the synthetic dataset
    derived from pytorch's Dataset class.
    """

    def __init__(self, root_path, patch_size=32, patch_num=5):
        self.root_dir_name = os.path.dirname(root_path)
        self.raw_dir_name = os.path.join(self.root_dir_name, "raw/")
        self.seg_dir_name = os.path.join(self.root_dir_name, "seg/")
        
        # Sets the patch size, and which patch to select
        self.patch_size = patch_size
        self.patch_num = patch_num

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            return [self[i] for i in range(index)]
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError("The index (%d) is out of range." % index)
            # get the data from direct index
            return self.get_item_from_index(index)
            
        else:
            raise TypeError("Invalid argument type.")
            
    def __len__(self):
        _, _, files = next(os.walk(self.raw_dir_name))
        return len(files)

    def get_item_from_index(self, index):
        raw_img_name = os.path.join(self.raw_dir_name, ("%d.nii.gz" % index))
        seg_img_name = os.path.join(self.seg_dir_name, ("%d.nii.gz" % index))
        
        # Load proxy so image not loaded into memory
        raw_proxy = nib.load(raw_img_name)
        seg_proxy = nib.load(seg_img_name)
        
        # Get dataobj of proxy
        raw_data = np.asarray(raw_proxy.dataobj).astype(np.int32)
        seg_data = np.asarray(seg_proxy.dataobj).astype(np.int32)

        # This is where the patch is define
        raw_patch = torch.from_numpy(raw_data).\
            unfold(2, self.patch_size, self.patch_size).\
            unfold(1, self.patch_size, self.patch_size).\
            unfold(0, self.patch_size, self.patch_size)
        raw_patch = raw_patch.contiguous().\
            view(-1, self.patch_size, self.patch_size, self.patch_size)

        seg_patch = torch.from_numpy(seg_data).\
            unfold(2, self.patch_size, self.patch_size).\
            unfold(1, self.patch_size, self.patch_size).\
            unfold(0, self.patch_size, self.patch_size)
        seg_patch = seg_patch.contiguous().\
            view(-1, self.patch_size, self.patch_size, self.patch_size)
        
        # Randomly return one patch?
        return raw_patch[self.patch_num], seg_patch[self.patch_num]
    
    