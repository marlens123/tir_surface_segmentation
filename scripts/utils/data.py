# Inspired by 2019 Pavel Iakubovskii https://github.com/qubvel/segmentation_models/

import numpy as np
from .preprocess_helpers import expand_greyscale_channels, get_training_augmentation, get_preprocessing, patch_extraction
from torch.utils.data import Dataset as BaseDataset
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import random
import os
import cv2

class Dataset(BaseDataset):
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
        mode,
        preprocessing,
        args=None,
        preprocessing_fn=get_preprocessing_fn(encoder_name="resnet34", pretrained="imagenet"),
        images = None,
        masks = None,
        im_size = None,
        num_classes = 3,
    ):
        self.mode = mode
        self.num_classes = num_classes

        if im_size is not None:
            self.im_size = im_size
        else:
            self.im_size = cfg_model["im_size"]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.classes]

        if images is not None and masks is not None:
            images, masks = patch_extraction(images, masks, size=self.im_size)
            self.images_fps = images.tolist()
            self.masks_fps = masks.tolist()
        else:
            if self.mode == "train":
                X_train, y_train = patch_extraction(np.load(args.path_to_X_train), np.load(args.path_to_y_train), size=self.im_size)
                self.images_fps = X_train.tolist()
                self.masks_fps = y_train.tolist()
            elif self.mode == "test":
                X_test, y_test = patch_extraction(np.load(args.path_to_X_test), np.load(args.path_to_y_test), size=self.im_size)
                self.images_fps = X_test.tolist()
                self.masks_fps = y_test.tolist()
            else:
                print("Specified mode must be either 'train' or 'test'")

        self.augmentation = cfg_training["augmentation"]
        self.augment_mode = cfg_training["augmentation_mode"]

        self.preprocessing = preprocessing
        self.preprocessing_fn = preprocessing_fn
        if "pretrain" in cfg_model:
            self.encoder_weights = cfg_model["pretrain"]
        else:
            self.encoder_weights = None

    def __getitem__(self, i):
        image = self.images_fps[i]
        # reshape to 3 dims in last channel
        image = expand_greyscale_channels(image)
        image = image.astype(np.float32)

        mask = self.masks_fps[i]
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(np.float32)

        if self.mode == "train" and self.augmentation:
            augmentation = get_training_augmentation(
                im_size=self.im_size, augment_mode=self.augment_mode
            )
            sample = augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            if self.encoder_weights in {"imagenet", "rsd46-whu", "aid", "satlas", "sam2_b+"}:
                print("Using imagenet preprocessing")
                image = self.preprocessing_fn(image)
            sample = self.preprocessing(image=image, mask=mask, pretraining=self.encoder_weights)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images_fps)


class PatchSamplingDataset(BaseDataset):
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
        mode,
        preprocessing,
        args=None,
        preprocessing_fn=get_preprocessing_fn(encoder_name="resnet34", pretrained="imagenet"),
        images = None,
        masks = None,
        patch_size = 128,
        oversample_prob = 0.5
    ):
        self.mode = mode
        self.im_size = cfg_model["im_size"]
        self.patch_size = patch_size
        self.oversample_prob = oversample_prob
        self.minority_class = 0 # melt_pond
        self.minority_coords = []

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.classes]

        if images is not None and masks is not None:
            images, masks = patch_extraction(images, masks, size=self.im_size)
            self.images_fps = images.tolist()
            self.masks_fps = masks.tolist()
        else:
            if self.mode == "train":
                # load normally, then find minority class pixels
                X_train, y_train = patch_extraction(np.load(args.path_to_X_train), np.load(args.path_to_y_train), size=self.im_size)

                for img_idx, mask in enumerate(y_train):
                    ys, xs = np.where(mask == self.minority_class)
                    for y, x in zip(ys, xs):
                        self.minority_coords.append((img_idx, y, x))

                self.images_fps = X_train.tolist()
                self.masks_fps = y_train.tolist()
            elif self.mode == "test":
                X_test, y_test = patch_extraction(np.load(args.path_to_X_test), np.load(args.path_to_y_test), size=self.im_size)
                self.images_fps = X_test.tolist()
                self.masks_fps = y_test.tolist()
            else:
                print("Specified mode must be either 'train' or 'test'")

        self.augmentation = cfg_training["augmentation"]
        self.augment_mode = cfg_training["augmentation_mode"]

        self.preprocessing = preprocessing
        self.preprocessing_fn = preprocessing_fn
        if "pretrain" in cfg_model:
            self.encoder_weights = cfg_model["pretrain"]
        else:
            self.encoder_weights = None

    def __getitem__(self, idx):
        if random.random() < self.oversample_prob and len(self.minority_coords) > 0:
            # pick pixel from the minority class
            img_idx, y, x = random.choice(self.minority_coords)
            img, mask = np.array(self.images_fps[img_idx]).astype(np.float32), np.array(self.masks_fps[img_idx]).astype(np.float32)

            # center patch around (y, x)
            y0 = max(0, min(y - self.patch_size // 2, img.shape[0] - self.patch_size))
            x0 = max(0, min(x - self.patch_size // 2, img.shape[1] - self.patch_size))

        else:
            # else sample a random patch
            img_idx = random.randrange(len(self.images_fps))
            img, mask = np.array(self.images_fps[img_idx]).astype(np.float32), np.array(self.masks_fps[img_idx]).astype(np.float32)
            y0 = random.randint(0, img.shape[0] - self.patch_size)
            x0 = random.randint(0, img.shape[1] - self.patch_size)

        # crop patch
        y1, x1 = y0 + self.patch_size, x0 + self.patch_size
        assert y1 <= img.shape[0] and x1 <= img.shape[1], f"Patch exceeds image boundaries with {y1}, {x1} vs {img.shape}, where y0={y0}, x0={x0}"
        patch_img = img[y0:y1, x0:x1]
        patch_mask = mask[y0:y1, x0:x1]

        # reshape to 3 dims in last channel
        patch_img = expand_greyscale_channels(patch_img)
        patch_mask = np.expand_dims(patch_mask, axis=-1)

        if self.mode == "train" and self.augmentation:
            augmentation = get_training_augmentation(
                im_size=self.patch_size, augment_mode=self.augment_mode
            )
            sample = augmentation(image=patch_img, mask=patch_mask)
            patch_img, patch_mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            if self.encoder_weights in {"imagenet", "rsd46-whu", "aid", "satlas", "sam2_b+"}:
                print("Using imagenet preprocessing")
                patch_img = self.preprocessing_fn(patch_img)
            sample = self.preprocessing(image=patch_img, mask=patch_mask, pretraining=self.encoder_weights)
            patch_img, patch_mask = sample["image"], sample["mask"]

        return patch_img, patch_mask

    def __len__(self):
        return len(self.images_fps)