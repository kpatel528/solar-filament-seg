"""Model factory. Same architecture for the from-scratch vs ImageNet-pretrained comparison."""
import segmentation_models_pytorch as smp


def build_model(encoder="resnet34", pretrained=True):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,          # grayscale H-alpha; smp adapts pretrained first-conv weights
        classes=1,
        decoder_attention_type="scse",
    )
