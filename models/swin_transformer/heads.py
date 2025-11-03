"""
Inspired by satlaspretrain_models/models/heads.py, adjusted to use ImageNet weights.
"""

import torch


class SimpleHead(torch.nn.Module):
    def __init__(self, task, backbone_channels, num_categories=2):
        super(SimpleHead, self).__init__()

        self.task_type = task

        use_channels = backbone_channels[0][1]
        num_layers = 2
        self.num_outputs = num_categories
        if self.num_outputs is None:
            if self.task_type == "regress":
                self.num_outputs = 1
            else:
                self.num_outputs = 2

        layers = []
        for _ in range(num_layers - 1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(use_channels, use_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)

        if self.task_type == "segment":
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            # modified here to ignore invalid pixels during the loss computation
            print("Using ignore_index=-1 for the loss function")
            self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(
                logits, targets, reduction="none", ignore_index=-1
            )

        else:
            raise NotImplementedError("Task type currently not supported.")

        self.layers = torch.nn.Sequential(*layers)

    # modified to only return raw logits
    def forward(self, image_list, raw_features, targets=None):
        outputs = self.layers(raw_features[0])

        return outputs
