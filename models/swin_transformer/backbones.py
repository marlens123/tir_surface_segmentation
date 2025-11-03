"""
Inspired by satlaspretrain_models/models/backbones.py, adjusted to use ImageNet weights.
"""

import torch.nn
import torchvision


class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch, weights="IMAGENET1K_V1"):
        super(SwinBackbone, self).__init__()

        print(f"Using {arch} backbone with {weights} weights.", flush=True)

        if arch == "swinb":
            self.backbone = torchvision.models.swin_v2_b(weights=weights)
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
        elif arch == "swint":
            self.backbone = torchvision.models.swin_v2_t(weights=weights)
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        else:
            raise ValueError("Backbone architecture not supported.")

        if num_channels != 3:
            print(
                f"Changing input channels from 3 to {num_channels}. Note that this layer won't be pretrained.",
                flush=True,
            )
            self.backbone.features[0][0] = torch.nn.Conv2d(
                num_channels,
                self.backbone.features[0][0].out_channels,
                kernel_size=(4, 4),
                stride=(4, 4),
            )

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]
