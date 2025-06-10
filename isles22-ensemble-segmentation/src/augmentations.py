import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=0, std=1, max_pixel_value=1.0),
        ToTensorV2()
    ])