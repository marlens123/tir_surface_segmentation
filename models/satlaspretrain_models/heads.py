"""
Copyright 2023 Bastani et al. https://github.com/allenai/satlaspretrain_models

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MODIFIED by marlens123 to return raw logits in SimpleHead forward pass.
"""


import collections
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from satlaspretrain_models.models.heads import SimpleHead as SatlasSimpleHead


class SimpleHead(SatlasSimpleHead):
    def __init__(self, task, backbone_channels, num_categories=2):
        super().__init__(task, backbone_channels, num_categories)

        self.task_type = task 

        use_channels = backbone_channels[0][1]
        num_layers = 2
        self.num_outputs = num_categories
        if self.num_outputs is None:
            if self.task_type == 'regress':
                self.num_outputs = 1
            else:
                self.num_outputs = 2

        layers = []
        for _ in range(num_layers-1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(use_channels, use_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)

        if self.task_type == 'segment':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        elif self.task_type == 'bin_segment':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            def loss_func(logits, targets):
                targets = targets.argmax(dim=1)
                return torch.nn.functional.cross_entropy(logits, targets, reduction='none')[:, None, :, :]
            self.loss_func = loss_func

        elif self.task_type == 'regress':
            layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))
            self.loss_func = lambda outputs, targets: torch.square(outputs - targets)

        elif self.task_type == 'classification':
            self.extra = torch.nn.Linear(use_channels, self.num_outputs)
            self.loss_func = lambda logits, targets: torch.nn.functional.cross_entropy(logits, targets, reduction='none')

        elif self.task_type == 'multi-label-classification':
            self.extra = torch.nn.Linear(use_channels, self.num_outputs)
            self.loss_func = lambda logits, targets: torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        self.layers = torch.nn.Sequential(*layers)

    # NOTE: modified by marlens123 to only return raw logits
    def forward(self, image_list, raw_features, targets=None):
        raw_outputs = self.layers(raw_features[0])

        return raw_outputs

