import segmentation_models_pytorch as smp

def get_unet(in_channels=9):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation='sigmoid'
    )