import numpy as np
import albumentations as A
from patchify import patchify

def expand_greyscale_channels(image):
    """
    Copies last channel three times to reach RGB-like shape.
    """
    image = np.expand_dims(image, -1)
    image = image.repeat(3, axis=-1)
    return image

def crop_center_square(image, im_size=480):
    """ "
    Crops the center of the input image with specified size.
    """
    size = im_size
    height, width = image.shape[:2]
    new_width = new_height = size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation(im_size=480, augment_mode=1, h=None, w=None):
    """
    structure inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb

    Defines augmentation for training data. Each technique applied with a probability.

    Parameters:
    -----------
        im_size : int
            size of the image

    Return:
    -------
        train_transform : albumentations.compose
    """

    if augment_mode == 0:
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),
            A.Rotate(interpolation=0),
            A.RandomSizedCrop(
                min_max_height=[int(0.5 * im_size), int(0.8 * im_size)],
                height=im_size,
                width=im_size,
                interpolation=0,
                p=0.5,
            ),
        ]

    if augment_mode == 1:
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.8,
            ),
            A.Rotate(interpolation=0),
            A.RandomSizedCrop(
                min_max_height=[int(0.5 * im_size), int(0.8 * im_size)],
                height=im_size,
                width=im_size,
                interpolation=0,
                p=0.5,
            ),
        ]

    if augment_mode == 2:
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),
            A.Rotate(interpolation=0),
        ]

    if augment_mode == 3:
        train_transform = [
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.8,
            ),
        ]

    if augment_mode == 4:
        train_transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.8,
            ),
        ]

    return A.Compose(train_transform)


def to_tensor(x, **kwargs):
    assert x.ndim == 3, f"Image must have 3 dims. Got {x.ndim} dims and shape {x.shape}."
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(pretraining, h=None, w=None):
    """Construct preprocessing transform.
    NOTE: normalization of imagenet, aid, rsd46-whu has already been done.

    Return:
        transform: albumentations.Compose

    """
    if pretraining == "imagenet":
        _transform = [
            A.Lambda(image=to_tensor, mask=to_tensor),
        ]
    # preprocessing according to https://github.com/lsh1994/remote_sensing_pretrained_models
    elif pretraining == "aidx":
        print("Using aid preprocessing")
        _transform = [
            A.Resize(height=224, width=224, p=1),
            #A.Normalize(p=1.0),
            A.Lambda(image=to_tensor, mask=to_tensor),
        ]
    elif pretraining == "rsd46-whux":
        print("Using rsd64-whu preprocessing")
        _transform = [
            A.Resize(height=256, width=256, p=1),
            A.Normalize(p=1.0),
            A.Lambda(image=to_tensor, mask=to_tensor),
        ]
    elif pretraining == "none" or pretraining is None or pretraining == "sa-1b":
        print("Using min max normalization")
        _transform = [
            A.Normalize(normalization="min_max", p=1.0),
            A.Lambda(image=to_tensor, mask=to_tensor),
        ]
    else:
        _transform = [
            A.Lambda(image=to_tensor, mask=to_tensor),
        ]
    return A.Compose(_transform)

def patch_extraction(imgs, masks, size):
    """
    Extracts patches from an image and mask using a sliding window with specified step size.
    """
    if size == 32:
        step = 32
    elif size == 64:
        step = 68
    elif size == 128:
        step = 160
    elif size == 256:
        step == 224
    elif size == 480:
        return imgs, masks
    else:
        print("Unknown patch size. Please enter 32, 64, 128, 256, 480.")

    img_patches = []
    for img in imgs:     
        patches_img = patchify(img, (size, size), step=step)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                img_patches.append(single_patch_img)
    images = np.array(img_patches)

    mask_patches = []
    for img in masks:
        patches_mask = patchify(img, (size, size), step=step)
        
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i,j,:,:]
                mask_patches.append(single_patch_mask)
    masks = np.array(mask_patches)

    return images, masks