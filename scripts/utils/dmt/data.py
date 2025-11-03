"""
Copyright (c) 2020, Zhengyang Feng
All rights reserved.

(BSD 3-Clause License)
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
“AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
OF THE POSSIBILITY OF SUCH DAMAGE.

MODIFIED by marlen123 to fit our data loading needs.
"""
import numpy as np
from ..preprocess_helpers import expand_greyscale_channels, get_training_augmentation, get_preprocessing, patch_extraction
from torch.utils.data import Dataset as BaseDataset
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import random
import os
import cv2


class SemiSupervisedDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        mode (str): Image mode ('train' or 'test')
    """

    CLASSES = ["melt_pond", "sea_ice", "ocean"]
    classes = ['melt_pond', 'sea_ice']

    def __init__(
        self,
        cfg_model,
        cfg_training,
        preprocessing,
        preprocessing_fn=get_preprocessing_fn(encoder_name="resnet34", pretrained="imagenet"),
        im_size = None,
        num_classes = 3,
        label_state = 0,    # 0: labeled, 1: unlabeled, 2: unlabeled
        image_set = None,
        mask_type = ".png",
        image_dir = "data/dmt/images/",
    ):
        
        assert image_set is not None, "image_set must be provided"
        assert mask_type in {".png", ".npy"}, "mask_type must be either .png or .npy"

        if image_set == 'test':
            assert label_state == 0, "Test set must be labeled"

        if label_state == 0:
            mask_dir = "data/dmt/masks/"
            assert "_unlabeled_" not in image_set, "Labeled set cannot have '_unlabeled_' in its name"
        else:
            mask_dir = "data/dmt/masks_pseudo/"  # dummy masks for unlabeled data
            assert "_labeled_" not in image_set, "Unlabeled set cannot have '_labeled_' in its name"

        if image_set == 'val' or image_set == 'test':
            image_dir = os.path.join(image_dir, image_set)
            mask_dir = os.path.join(mask_dir, image_set)
            self.mode = "test"
            file_names_images = sorted([x for x in os.listdir(image_dir) if x.endswith(".png") or x.endswith(".npy")])
            file_names_images = [x.replace(".png", "").replace(".npy", "") for x in file_names_images]
            file_names_masks = sorted([x for x in os.listdir(mask_dir) if x.endswith(".png") or x.endswith(".npy")])
            file_names_masks = [x.replace(".png", "").replace(".npy", "") for x in file_names_masks]
        elif label_state == 0:
            image_dir = os.path.join(image_dir, 'train', image_set)
            mask_dir = os.path.join(mask_dir, 'train', image_set)
            self.mode = "train"
            file_names_images = sorted([x for x in os.listdir(image_dir) if x.endswith(".png") or x.endswith(".npy")])
            file_names_images = [x.replace(".png", "").replace(".npy", "") for x in file_names_images]
            file_names_masks = sorted([x for x in os.listdir(mask_dir) if x.endswith(".png") or x.endswith(".npy")])
            file_names_masks = [x.replace(".png", "").replace(".npy", "") for x in file_names_masks]
        else:
            image_dir = os.path.join(image_dir, 'train', image_set)
            mask_dir = os.path.join(mask_dir, 'train', image_set)
            self.mode = "train"

            # We first generate data lists before all this, so we can do this easier
            splits_dir = "data/dmt/sets_dir/"
            split_f = os.path.join(splits_dir, image_set + '.txt')
            with open(os.path.join(split_f), "r") as f:
                file_names = [x.strip() for x in f.readlines()]
                # remove file extension if present
                file_names = [x.replace(".png", "").replace(".npy", "").replace("/", "_") for x in file_names]
            file_names_images = file_names
            file_names_masks = file_names  # dummy masks for unlabeled data

        self.images = [os.path.join(image_dir, x + ".png") for x in file_names_images]
        self.masks = [os.path.join(mask_dir, x + mask_type) for x in file_names_masks]

        if label_state == 2:
            self.has_label = False
        else:
            self.has_label = True

        self.num_classes = num_classes
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.classes]
        if im_size is not None:
            self.im_size = im_size
        else:
            self.im_size = cfg_model["im_size"]

        self.augmentation = cfg_training["augmentation"]
        self.augment_mode = cfg_training["augmentation_mode"]
        self.preprocessing = preprocessing
        self.preprocessing_fn = preprocessing_fn
        if "pretrain" in cfg_model:
            self.encoder_weights = cfg_model["pretrain"]
        else:
            self.encoder_weights = None

    def __getitem__(self, index):
        img = cv2.imread(self.images[index], 0)  # read as greyscale
        assert img.shape == (self.im_size, self.im_size), f"Image shape: {img.shape}, expected ({self.im_size}, {self.im_size})"
        img = expand_greyscale_channels(img).astype(np.float32)
        assert img.shape == (self.im_size, self.im_size, 3), f"Image shape: {img.shape}, expected ({self.im_size}, {self.im_size}, 3)"
        
        if self.has_label:
            # Return x (input image) & y (mask images as a list)
            # Supports .png & .npy
            target = cv2.imread(self.masks[index], 0) if '.png' in self.masks[index] else np.load(self.masks[index])
            if not target.ndim == 3:
                target = np.expand_dims(np.array(target), axis=-1).astype(np.float32)

            if self.mode == "train" and self.augmentation:
                augmentation = get_training_augmentation(
                    im_size=self.im_size, augment_mode=self.augment_mode
                )
                sample = augmentation(image=img, mask=target)
                img, target = sample["image"], sample["mask"]

            # apply preprocessing
            if self.preprocessing:
                if self.encoder_weights in {"imagenet", "rsd46-whu", "aid", "satlas", "sam2_b+"}:
                    print("Using imagenet preprocessing")
                    img = self.preprocessing_fn(img)
                sample = self.preprocessing(image=img, mask=target, pretraining=self.encoder_weights)
                img, target = sample["image"], sample["mask"]

            return img, target
        else:
            # Return x (input image) & filenames & original image size as a list to store pseudo label
            target = self.masks[index]

            # apply preprocessing
            if self.preprocessing:
                if self.encoder_weights in {"imagenet", "rsd46-whu", "aid", "satlas", "sam2_b+"}:
                    print("Using imagenet preprocessing")
                    img = self.preprocessing_fn(img)
                sample = self.preprocessing(image=img, pretraining=self.encoder_weights)
                img = sample["image"]
            return img, target

    def __len__(self):
        return len(self.images)