"""
Inspired by satlaspretrain_models/model.py, adjusted to use ImageNet weights.
"""

from .heads import SimpleHead
from .fpn import FPN, Upsample
from .backbones import SwinBackbone
from .utils import Backbone, Head
import torch


class ImageNetWeights:
    def __init__(self):
        """
        Class to manage downloading weights and formatting them to be loaded into SatlasPretrain models.
        """
        super(ImageNetWeights, self).__init__()

    def get_pretrained_model(
        self,
        backbone="swinb",
        weights="IMAGENET1K_V1",
        fpn=False,
        head=None,
        num_categories=None,
        device="cuda",
    ):
        """
        Find and load pretrained SatlasPretrain weights, based on the model_identifier argument.
        Option to load pretrained FPN and/or a randomly initialized head.

        Args:
            model_identifier:
            fpn (bool): Whether or not to load a pretrained FPN along with the Backbone.
            head (enum): If specified, a randomly initialized Head will be created along with the
                        Backbone and [optionally] the FPN.
            num_categories (int): Number of categories to be included in output from prediction head.
        """
        if head and (num_categories is None):
            raise ValueError("Must specify num_categories if head is desired.")

        if weights not in ["IMAGENET1K_V1", None]:
            raise ValueError(
                "Currently only IMAGENET1K_V1 weights or training from scratch is supported."
            )

        if backbone == "swinb":
            backbone = Backbone.SWINB
        elif backbone == "swint":
            backbone = Backbone.SWINT
        else:
            raise ValueError("Currently only SWINB and SWINT backbones are supported.")

        # Initialize a pretrained model using the Model() class.
        model = Model(
            backbone=backbone,
            fpn=fpn,
            head=head,
            num_categories=num_categories,
            weights=weights,
        )
        return model


class Model(torch.nn.Module):
    def __init__(
        self,
        num_channels=3,
        multi_image=False,
        backbone=Backbone.SWINB,
        fpn=False,
        head=None,
        num_categories=None,
        weights=None,
    ):
        """
        Initializes a model, based on desired imagery source and model components. This class can be used directly to
        create a randomly initialized model (if weights=None) or can be called from the Weights class to initialize a
        SatlasPretrain pretrained foundation model.

        Args:
            num_channels (int): Number of input channels that the backbone model should expect.
            multi_image (bool): Whether or not the model should expect single-image or multi-image input.
            backbone (Backbone): The architecture of the pretrained backbone. All image sources support SwinTransformer.
            fpn (bool): Whether or not to feed imagery through the pretrained Feature Pyramid Network after the backbone.
            head (Head): If specified, a randomly initialized head will be included in the model.
            num_categories (int): If a Head is being returned as part of the model, must specify how many outputs are wanted.
            weights (torch weights): Weights to be loaded into the model. Defaults to None (random initialization) unless
                                    initialized using the Weights class.
        """
        super(Model, self).__init__()

        # Validate user-provided arguments.
        if not isinstance(backbone, Backbone):
            raise ValueError(f"Invalid backbone: {backbone}.")
        if head and not isinstance(head, Head):
            raise ValueError(f"Invalid head: {head}.")
        if head and (num_categories is None):
            raise ValueError("Must specify num_categories if head is desired.")

        self.backbone = self._initialize_backbone(
            num_channels, backbone, multi_image, weights
        )

        if fpn:
            self.fpn = self._initialize_fpn(self.backbone.out_channels)
            self.upsample = Upsample(self.fpn.out_channels)
        else:
            self.fpn = None

        if head:
            self.head = (
                self._initialize_head(head, self.fpn.out_channels, num_categories)
                if fpn
                else self._initialize_head(
                    head, self.backbone.out_channels, num_categories
                )
            )
        else:
            self.head = None

    def _initialize_backbone(self, num_channels, backbone_arch, multi_image, weights):
        # Load backbone model according to specified architecture.
        if backbone_arch == Backbone.SWINB:
            backbone = SwinBackbone(num_channels, arch="swinb", weights=weights)
        elif backbone_arch == Backbone.SWINT:
            backbone = SwinBackbone(num_channels, arch="swint", weights=weights)
        else:
            raise ValueError("Unsupported backbone architecture.")

        # If using a model for multi-image, need the Aggretation to wrap underlying backbone model.
        prefix, prefix_allowed_count = None, None
        if backbone_arch in [Backbone.RESNET50, Backbone.RESNET152]:
            prefix_allowed_count = 0
        elif multi_image:
            raise ValueError(
                "Multi-image not supported for this backbone architecture."
            )
        else:
            prefix_allowed_count = 1

        """
        # Load pretrained weights into the intialized backbone if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(weights, 'backbone', 'backbone.', prefix_allowed_count)
            backbone.load_state_dict(state_dict)"
        """

        return backbone

    # modified: pretrained weights are used in the backbone, so we don't need to load them again in the fpn
    def _initialize_fpn(self, backbone_channels):
        fpn = FPN(backbone_channels)
        return fpn

    def _initialize_head(self, head, backbone_channels, num_categories):
        # Initialize the head (classification, detection, etc.) if specified
        if head == Head.SEGMENT:
            return SimpleHead("segment", backbone_channels, num_categories)
        else:
            raise NotImplementedError("Head type currently not supported.")
        return None

    def forward(self, imgs, targets=None):
        # Define forward pass
        x = self.backbone(imgs)
        if self.fpn:
            x = self.fpn(x)
            x = self.upsample(x)
        if self.head:
            x = self.head(imgs, x, targets)
            return x
        return x
