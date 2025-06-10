import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

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
        dwi = self.load_nifti(sample["dwi"])    # [H, W, D]
        flair = self.load_nifti(sample["flair"])  # [H, W, D]
    
        # Crop both to the minimum shape
        min_shape = np.minimum(dwi.shape, flair.shape)
        dwi_cropped = dwi[:min_shape[0], :min_shape[1], :min_shape[2]]
        flair_cropped = flair[:min_shape[0], :min_shape[1], :min_shape[2]]
    
        x = np.stack([dwi_cropped, flair_cropped], axis=0)  # [2, H, W, D]
        y = self.load_nifti(sample["mask"])
        y = y[:min_shape[0], :min_shape[1], :min_shape[2]]  # Crop mask to match
    
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    @staticmethod
    def load_nifti(path):
        return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)