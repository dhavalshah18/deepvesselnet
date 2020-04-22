import os
import numpy as np
import torch
import torch.utils.data as data
# from torchvision import transforms
import nibabel as nib

from dvn import patchify_unpatchify as pu
from dvn import transforms


class SyntheticData(data.Dataset):
    """ 
    Class defined to handle the synthetic dataset
    derived from pytorch's Dataset class.
    """

    def __init__(self, root_path, patch_size=64, mode="train"):
        self.root_dir_name = os.path.dirname(root_path)
        self.raw_dir_name = os.path.join(self.root_dir_name, "raw/")
        self.seg_dir_name = os.path.join(self.root_dir_name, "seg/")
        self.mode = mode

        # Sets the patch size, and which patch to select
        self.patch_size = patch_size
        
        # Finding first and last file names in directory
        file_names = os.listdir(self.raw_dir_name)
        file_names.sort(key=lambda name: int(name.replace(".nii.gz", "")))
        
        first = file_names[0]
        self.first_file_num = int(first.replace(".nii.gz", ""))
        
        last = file_names[-1]
        self.last_file_num = int(last.replace(".nii.gz", ""))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            return [self[i] for i in range(index)]
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < self.first_file_num:
                index += self.first_file_num
            if index > self.last_file_num:
                index -= self.last_file_num
            if index < self.first_file_num or index > self.last_file_num:
                raise IndexError("The index (%d) is out of range." % index)
            
            # get the data from direct index
            return self.get_item_from_index(index)

        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return self.last_file_num - self.first_file_num + 1

    def get_item_from_index(self, index):
        raw_img_name = os.path.join(self.raw_dir_name, ("%d.nii.gz" % index))
        seg_img_name = os.path.join(self.seg_dir_name, ("%d.nii.gz" % index))
        
        # Load proxy so image not loaded into memory
        raw_proxy = nib.load(raw_img_name)
        seg_proxy = nib.load(seg_img_name)

        # Get dataobj of proxy
        raw_data = np.asarray(raw_proxy.dataobj).astype(np.int32)
        seg_data = np.asarray(seg_proxy.dataobj).astype(np.int32)
        
        if self.mode == "test":
            return torch.from_numpy(raw_data), torch.from_numpy(seg_data)
        
        # This is where the patch is defined
        raw_patch = torch.from_numpy(raw_data). \
            unfold(2, self.patch_size, self.patch_size). \
            unfold(1, self.patch_size, self.patch_size). \
            unfold(0, self.patch_size, self.patch_size)
        raw_patch = raw_patch.contiguous(). \
            view(-1, 1, self.patch_size, self.patch_size, self.patch_size)

        seg_patch = torch.from_numpy(seg_data). \
            unfold(2, self.patch_size, self.patch_size). \
            unfold(1, self.patch_size, self.patch_size). \
            unfold(0, self.patch_size, self.patch_size)
        seg_patch = seg_patch.contiguous(). \
            view(-1, self.patch_size, self.patch_size, self.patch_size)
        
        # Random number to select patch number
        patch_num = np.random.randint(raw_patch.shape[0])
        
        

        # Randomly return one patch
        return raw_patch[patch_num], seg_patch[patch_num]
    
    
class MRAData(data.Dataset):
    """ 
    Class defined to handle MRA data
    derived from pytorch's Dataset class.
    """
    
    def __init__(self, root_path, transform="normalize", 
                 patch_size=64, stride=60, mode="train"):
        self.root_dir_name = os.path.dirname(root_path)
        self.raw_dir_name = os.path.join(self.root_dir_name, "raw/")
        self.seg_dir_name = os.path.join(self.root_dir_name, "seg/")
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.name = ""
        
        self.file_names = os.listdir(self.raw_dir_name)
        
    def __len__(self):
        _, _, files = next(os.walk(self.raw_dir_name))
        return len(files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            return [self[i] for i in range(index)]
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index > len(self):
                raise IndexError("The index (%d) is out of range." % index)
            
            # get the data from direct index
            return self.get_item_from_index(index)

        else:
            raise TypeError("Invalid argument type.")
    
    def get_item_from_index(self, index):
        name = self.file_names[index]
        raw_img_name = os.path.join(self.raw_dir_name, name)
        self.name = raw_img_name
        seg_img_name = os.path.join(self.seg_dir_name, name)
        
        # Load proxy so image not loaded into memory
        raw_proxy = nib.load(raw_img_name)
        seg_proxy = nib.load(seg_img_name)

        # Get dataobj of proxy
        raw_data = np.asarray(raw_proxy.dataobj).astype(np.int32)
        seg_data = np.asarray(seg_proxy.dataobj).astype(np.int32)
#         print("Num of seg pixels: ", np.argwhere(seg_data == 1).size)

        raw_image = torch.from_numpy(raw_data)
        
        seg_image = torch.from_numpy(seg_data)
        
        # If training only return single patch
        if self.mode == "train":
            raw_patch = raw_image. \
                unfold(2, self.patch_size, self.patch_size). \
                unfold(1, self.patch_size, self.patch_size). \
                unfold(0, self.patch_size, self.patch_size)
            raw_patch = raw_patch.contiguous(). \
                view(-1, 1, self.patch_size, self.patch_size, self.patch_size)

            seg_patch = seg_image. \
                unfold(2, self.patch_size, self.patch_size). \
                unfold(1, self.patch_size, self.patch_size). \
                unfold(0, self.patch_size, self.patch_size)
            seg_patch = seg_patch.contiguous(). \
                view(-1, self.patch_size, self.patch_size, self.patch_size)
            
            # Random number to select patch number
            patch_num = np.random.randint(seg_patch.shape[0])
            
            raw_image = raw_patch[patch_num]
            seg_image = seg_patch[patch_num]
            
        
        if self.transform == "normalize":
            normalize = transforms.Normalize(torch.max(raw_image), torch.min(raw_image), 0., 255.)
            raw_image = normalize(raw_image)
        
        return raw_image, seg_image
    
    