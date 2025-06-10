import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import cv2


class ISLES2p5DDataset(Dataset):
    def __init__(self, root_dir, modalities=['dwi', 'adc', 'flair'], mask_name='mask', 
                 slice_axis=2, slice_depth=3, transform=None, resize=None):
        self.root_dir = root_dir
        self.modalities = modalities
        self.mask_name = mask_name
        self.slice_axis = slice_axis
        self.slice_depth = slice_depth
        self.transform = transform
        self.resize = resize
        self.samples = []

        for subject in os.listdir(self.root_dir):
            if not subject.startswith("sub-"):
                continue
            ses_dir = os.path.join(self.root_dir, subject, "ses-0001")
            if not os.path.exists(ses_dir):
                continue
            mod_paths = {}
            for mod in self.modalities:
                mod_dir = os.path.join(ses_dir, "dwi" if mod in ["dwi", "adc"] else "anat")
                files = [f for f in os.listdir(mod_dir)] if os.path.exists(mod_dir) else []
                found = [f for f in files if mod in f.lower() and f.endswith(".nii.gz")]
                if found:
                    mod_paths[mod] = os.path.join(mod_dir, found[0])
            mask_dir = os.path.join(self.root_dir, "derivatives", subject, "ses-0001")
            mask_files = [f for f in os.listdir(mask_dir)] if os.path.exists(mask_dir) else []
            mask_found = [f for f in mask_files if f.endswith(".nii.gz")]
            if mask_found:
                mod_paths['mask'] = os.path.join(mask_dir, mask_found[0])
            if all(m in mod_paths for m in self.modalities) and 'mask' in mod_paths:
                # Get the minimum number of slices across all modalities and mask
                num_slices_list = []
                for mod in self.modalities + ['mask']:
                    img = nib.load(mod_paths[mod]).get_fdata()
                    num_slices_list.append(img.shape[self.slice_axis])
                min_slices = min(num_slices_list)
                pad = self.slice_depth // 2
                for idx in range(pad, min_slices - pad):
                    self.samples.append((mod_paths.copy(), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
    mod_paths, slice_idx = self.samples[idx]
    imgs = []
    target_shape = self.resize if self.resize is not None else None  # e.g., (128, 128)
    # Stack slices for each modality
    for mod in self.modalities:
        img_3d = nib.load(mod_paths[mod]).get_fdata()
        img_3d = (img_3d - img_3d.mean()) / (img_3d.std() + 1e-5)
        pad = self.slice_depth // 2
        slices = []
        for offset in range(-pad, pad+1):
            slc = np.take(img_3d, slice_idx+offset, axis=self.slice_axis)
            if target_shape is not None:
                slc = cv2.resize(slc, target_shape[::-1], interpolation=cv2.INTER_LINEAR)
            slices.append(slc)
        imgs.extend(slices)
    x = np.stack(imgs, axis=0).astype(np.float32)
    # Mask (single slice, center)
    mask_3d = nib.load(mod_paths['mask']).get_fdata()
    mask = np.take(mask_3d, slice_idx, axis=self.slice_axis)
    mask = (mask > 0).astype(np.float32)
    if target_shape is not None:
        mask = cv2.resize(mask, target_shape[::-1], interpolation=cv2.INTER_NEAREST)
    if self.transform:
        augmented = self.transform(image=x.transpose(1,2,0), mask=mask)
        x = augmented['image']
        mask = augmented['mask']
    else:
        x = torch.tensor(x)
        mask = torch.tensor(mask)
    return x, mask