# Costum adaptations to https://github.com/qubvel-org/segmentation_models.pytorch/tree/main to allow for custom weights to be loaded.

from models.smp.base_model_smp import SegmentationModel as SegmentationModel_Features
from segmentation_models_pytorch.base.heads import SegmentationHead, ClassificationHead
from segmentation_models_pytorch.decoders.unet.model import Unet as Unet_SMP
from segmentation_models_pytorch.decoders.pspnet.model import PSPNet as PSPNet_SMP
from segmentation_models_pytorch.decoders.deeplabv3.model import DeepLabV3Plus as DeepLabV3Plus_SMP
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

import warnings


import torch
from segmentation_models_pytorch.encoders import encoders
from segmentation_models_pytorch.encoders import TimmUniversalEncoder
import torch.utils.model_zoo as model_zoo

from typing import Optional as _Optional
from typing import Union as _Union
from typing import List
import torch as _torch

def get_encoder_rs(name, in_channels=3, depth=5, weights=None, output_stride=32, pretrain=None, **kwargs):
    """
    Function adapted from segmentation_models_pytorch.encoders.get_encoder to allow for loading of custom weights
    """
    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs,
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError(
            "Wrong encoder name `{}`, supported encoders: {}".format(
                name, list(encoders.keys())
            )
        )

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if pretrain == "rsd46-whu":
        print("using rsd weights")
        weights = torch.load(r"pretraining_checkpoints/resnet34/{}/resnet34-epoch.19-val_acc.0.921.ckpt".format(pretrain))["state_dict"]
        
        for k in list(weights.keys()):
            weights[str(k)[4:]]=weights.pop(k)

        weights.pop("fc.0.bias", None)
        weights.pop("fc.0.weight", None)
        encoder.load_state_dict(weights)

    elif pretrain == "aid":
        print("using aid weights")
        weights = torch.load(r"pretraining_checkpoints/resnet34/{}/resnet34_224-epoch.9-val_acc.0.966.ckpt".format(pretrain))["state_dict"]
        for k in list(weights.keys()):
            weights[str(k)[4:]]=weights.pop(k)

        weights.pop("fc.0.bias", None)
        weights.pop("fc.0.weight", None)
        encoder.load_state_dict(weights)

    elif weights is not None:
        print("using imagenet weights")
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, name, list(encoders[name]["pretrained_settings"].keys())
                )
            )
        weights = model_zoo.load_url(settings["url"])
        print(weights.keys())
        encoder.load_state_dict(weights)
        encoder.set_in_channels(in_channels, pretrained=weights is not None)

    if output_stride != 32:
        encoder.make_dilated(output_stride)
    print(encoder)

    return encoder

class Unet(Unet_SMP):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: _Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: _Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: _Optional[_Union[str, callable]] = None,
        aux_params: _Optional[dict] = None,
        pretrain: _Optional[str] = None,
        **kwargs
    ):
        super().__init__(encoder_name, encoder_depth, encoder_weights, decoder_use_batchnorm, decoder_channels, decoder_attention_type, in_channels, classes, activation, aux_params, **kwargs)

        self.encoder = get_encoder_rs(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            pretrain=pretrain,
        )

# NOTE: Took UnetPlusPlus class completely from SMP to be able to inherit from our SegmentationModel_Features
# adjusted the encoder to use our get_encoder_rs function
class UnetPlusPlus(SegmentationModel_Features):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: _Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: _Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: _Optional[_Union[str, callable]] = None,
        aux_params: _Optional[dict] = None,
        pretrain: _Optional[str] = None,
        get_features: bool = False,
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError(
                "UnetPlusPlus is not support encoder_name={}".format(encoder_name)
            )

        self.encoder = get_encoder_rs(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            pretrain=pretrain,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()

class PSPNet(PSPNet_SMP):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: _Optional[str] = "imagenet",
        encoder_depth: int = 3,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
        in_channels: int = 3,
        classes: int = 1,
        activation: _Optional[_Union[str, callable]] = None,
        upsampling: int = 8,
        aux_params: _Optional[dict] = None, 
        pretrain: _Optional[str] = None, 
        **kwargs):

        super().__init__(encoder_name, encoder_weights, encoder_depth, psp_out_channels, psp_use_batchnorm, psp_dropout, in_channels, classes, activation, upsampling, aux_params, **kwargs)

        self.encoder = get_encoder_rs(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            pretrain=pretrain,
        )

class DeepLabV3Plus(DeepLabV3Plus_SMP):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: _Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: _Optional[str] = None,
        upsampling: int = 4,
        aux_params: _Optional[dict] = None,
        pretrain: _Optional[str] = None,
        **kwargs
    ):

        super().__init__(encoder_name, encoder_depth, encoder_weights, encoder_output_stride, decoder_channels, decoder_atrous_rates, in_channels, classes, activation, upsampling, aux_params, **kwargs)

        self.encoder = get_encoder_rs(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            pretrain=pretrain,
        )

def create_model_rs(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: _Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    pretrain: _Optional[str] = "imagenet",
    get_features: bool = False,
    **kwargs,
) -> _torch.nn.Module:
    """
    Function adapted from https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/__init__.py
    """
    archs = [
        Unet,
        UnetPlusPlus,
        PSPNet,
        DeepLabV3Plus,
    ]
    archs_dict = {a.__name__: a for a in archs}
    try:
        model_class = archs_dict[arch]
    except KeyError:
        raise KeyError(
            "Wrong architecture type `{}`. Available options are: {}".format(
                arch, list(archs_dict.keys())
            )
        )
    
    if get_features: 
        return model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            pretrain=pretrain,
            get_features=get_features,
            **kwargs,
        )
    else:
        return model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            pretrain=pretrain,
            **kwargs,
        )