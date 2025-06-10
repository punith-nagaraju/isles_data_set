import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import cv2

class ISLESDataset3D(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        print(f"entering 3D samples")
        mask_root = os.path.join(root_dir, "derivatives")
        for subject in os.listdir(root_dir):
            if subject.startswith("sub-"):
                ses_dir = os.path.join(root_dir, subject, "ses-0001")
                if os.path.exists(ses_dir):
                    dwi_dir = os.path.join(ses_dir, "dwi")
                    anat_dir = os.path.join(ses_dir, "anat")
                    dwi_path = [f for f in os.listdir(dwi_dir) if f.endswith("_dwi.nii.gz")]
                    flair_path = [f for f in os.listdir(anat_dir) if f.endswith("_FLAIR.nii.gz")]
                    mask_dir = os.path.join(mask_root, subject, "ses-0001")
                    mask_path = []
                    if os.path.exists(mask_dir):
                        mask_path = [f for f in os.listdir(mask_dir) if f.endswith(".nii.gz")]
                    if dwi_path and flair_path and mask_path:
                        self.samples.append({
                            "dwi": os.path.join(dwi_dir, dwi_path[0]),
                            "flair": os.path.join(anat_dir, flair_path[0]),
                            "mask": os.path.join(mask_dir, mask_path[0])
                        })
        print(f"Total 3D samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dwi = nib.load(sample["dwi"]).get_fdata()
        flair = nib.load(sample["flair"]).get_fdata()
        mask = nib.load(sample["mask"]).get_fdata()
        # Crop to minimum shape
       # After normalization, before stacking:
        crop_shape = (64, 64, 16)  # or smaller if needed
        dwi = dwi[:crop_shape[0], :crop_shape[1], :crop_shape[2]]
        flair = flair[:crop_shape[0], :crop_shape[1], :crop_shape[2]]
        mask = mask[:crop_shape[0], :crop_shape[1], :crop_shape[2]]
        # Pad to be divisible by 16
        def pad_to_16(arr):
            pad = [(0, (16 - s % 16) % 16) for s in arr.shape]
            return np.pad(arr, pad, mode='constant')
        dwi = pad_to_16(dwi)
        flair = pad_to_16(flair)
        mask = pad_to_16(mask)
        # Normalize
        dwi = (dwi - dwi.mean()) / (dwi.std() + 1e-5)
        flair = (flair - flair.mean()) / (flair.std() + 1e-5)
        x = np.stack(imgs, axis=0).astype(np.float32)  # [channels, H, W]
        y = (mask > 0).astype(np.float32)
         if self.transform:
        augmented = self.transform(image=x.transpose(1,2,0), mask=mask)
        x = augmented['image']  # [channels, H, W] if ToTensorV2 is used
        mask = augmented['mask']
        else:
            x = torch.tensor(x)
            mask = torch.tensor(mask)
        return x, mask

