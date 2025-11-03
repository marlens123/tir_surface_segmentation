from models.AutoSAM.models.build_autosam_seg_model import sam_seg_model_registry as autosam
import torch.nn.functional as F
import numpy as np
import cv2
import torch
from .preprocess_helpers import expand_greyscale_channels, crop_center_square, get_preprocessing
import segmentation_models_pytorch as smp
from models.smp.build_rs_models import create_model_rs

def label_to_pixelvalue_with_uncertainty(image):
    """
    Transforms class labels to pixelvalues in the grayscale range to be able to make outcomes visible.
    """
    uniques = np.unique(image)

    for idx, elem in enumerate(uniques):
        mask = np.where(image == 1)
        image[mask] = 85
        mask2 = np.where(image == 2)
        image[mask2] = 170
        mask3 = np.where(image == 3)
        image[mask3] = 255
    return image

def label_to_pixelvalue(image):
    """
    Transforms class labels to pixelvalues in the grayscale range to be able to make outcomes visible.
    """
    uniques = np.unique(image)

    for idx, elem in enumerate(uniques):
        mask = np.where(image == 1)
        image[mask] = 125
        mask2 = np.where(image == 2)
        image[mask2] = 255
    return image

def preprocess_prediction(image, model_preprocessing, normalize=False, preprocessing_fn=None):
    """
    Preprocesses image to be suitable as input for model prediction.
    """
    image = expand_greyscale_channels(image)

    # create mask of zeros such that preprocessing function works
    random_mask = np.zeros(image.shape)

    image = image.astype(np.float32)

    # apply normalization
    if normalize:
        print("Normalizing image")
        image = (image - image.mean()) / image.std()

    if preprocessing_fn is not None:
        print("Using backbone preprocessing function")
        image = preprocessing_fn(image)

    sample = model_preprocessing(image=image, mask=random_mask)
    image, _ = sample["image"], sample["mask"]

    # will add a dimension that replaces batch_size
    image = np.expand_dims(image, axis=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(image, device=device)

    return image

def autosam_patch_predict(model, image, patch_size, model_preprocessing, normalize):
    """
    Predicts on image patches and recombines masks to whole image later.

    This function is inspired by
    https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py
    """
    # initialize mask with zeros
    segm_img = np.zeros(image.shape[:2])
    patch_num = 1
    image_probabilities = np.zeros(image.shape[:2])
    # Iterates through image in steps of patch_size, operates on patches
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            single_patch = image[i : i + patch_size, j : j + patch_size]
            single_patch_shape = single_patch.shape[:2]
            single_patch = preprocess_prediction(
                single_patch, model_preprocessing=model_preprocessing, normalize=normalize
            )
            mask, _ = model.forward(single_patch)
            probabilities = F.softmax(mask, dim=0)
            probabilities = np.array(probabilities.cpu().detach().squeeze())
            mask = np.array(mask.cpu().detach())
            mask = np.argmax(mask.squeeze(axis=1), axis=0)

            # recombine to complete image
            segm_img[i : i + single_patch_shape[0], j : j + single_patch_shape[1]] += (
                cv2.resize(mask, single_patch_shape[::-1])
            )
            image_probabilities[i : i + single_patch_shape[0], j : j + single_patch_shape[1]] += (
                cv2.resize(probabilities, single_patch_shape[::-1])
            )
            print("Finished processing patch number ", patch_num, " at position ", i, j)
            patch_num += 1

    return mask, image_probabilities

def predict_image_autosam(
    img,
    im_size,
    weights,
    pretraining="sa-1b",
    autosam_size="vit_b",
    save_path=None,
    normalize=False,
    no_finetune=False,        
):
    """
    Preprocesses image for prediction, loads model with weights and uses model to predict segmentation mask.
    """

    if autosam_size == "vit_h":
        model_checkpoint = "pretraining_checkpoints/segment_anything/sam_vit_h_4b8939.pth"
    elif autosam_size == "vit_l":
        model_checkpoint = "pretraining_checkpoints/segment_anything/sam_vit_l_0b3195.pth"
    elif autosam_size == "vit_b":
        model_checkpoint = "pretraining_checkpoints/segment_anything/sam_vit_b_01ec64.pth"
    
    model = autosam[autosam_size](num_classes=3, checkpoint=model_checkpoint)
    model = model.cuda(0)

    preprocessing = get_preprocessing(pretraining=pretraining)

    # load model weights
    if not no_finetune:
        model.load_state_dict(torch.load(weights))

    # crop the image to be predicted to a size that is divisible by the patch size used
    if im_size == 480:
        preprocessed = preprocess_prediction(
            img, model_preprocessing=preprocessing, normalize=normalize
        )
        mask, _ = model.forward(preprocessed)
        print(mask.shape)
        probabilities = F.softmax(mask, dim=0)
        probabilities = np.array(probabilities.cpu().detach().squeeze())
        mask = np.array(mask.cpu().detach())
        mask = np.argmax(mask.squeeze(axis=1), axis=0)
    elif im_size == 256:
        img = crop_center_square(img, 256)
        mask, probabilities = autosam_patch_predict(model=model, image=img, patch_size=256, model_preprocessing=preprocessing, normalize=normalize)
    elif im_size == 128:
        img = crop_center_square(img, 384)
        mask, probabilities = autosam_patch_predict(model=model, image=img, patch_size=128, model_presprocessing=preprocessing, normalize=normalize)
    elif im_size == 64:
        img = crop_center_square(img, 448)
        mask, probabilities = autosam_patch_predict(model=model, image=img, patch_size=448, model_preprocessing=preprocessing, normalize=normalize)

    if save_path is not None:
        cv2.imwrite(save_path, mask)
    return mask, probabilities


def predict_image_smp(
    arch,
    img,
    im_size,
    weights,
    pretraining="imagenet",
    save_path=None,
    normalize=False,
    no_finetune=False
):
    # NOTE: we can load imagenet weights here because we will overwrite them with our fine-tuned weights right after
    model = create_model_rs(
        arch=arch,
        encoder_name="resnet34",
        pretrain="aid",
        in_channels=3,
        classes=3,
        get_features=False
    )
    model = model.cuda(0)

    # load model weights
    if not no_finetune:
        model.load_state_dict(torch.load(weights))

    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet34", pretrained="imagenet")
    preprocessing = get_preprocessing(pretraining=pretraining)

    # crop the image to be predicted to a size that is divisible by the patch size used
    if im_size == 480:
        preprocessed = preprocess_prediction(
            img, model_preprocessing=preprocessing, normalize=normalize, preprocessing_fn=preprocessing_fn
        )
        mask = model.forward(preprocessed)
        probabilities = F.softmax(mask, dim=1)
        probabilities = np.array(probabilities.cpu().detach().squeeze())
        mask = np.array(mask.cpu().detach())
        mask = np.argmax(mask.squeeze(0), axis=0)
    else: 
        raise NotImplementedError("Only 480x480 images are supported for now.")
    
    if save_path is not None:
        cv2.imwrite(save_path, mask)
    return mask, probabilities

def extract_features(
    arch,
    img,
    im_size,
    weights,
    pretraining="imagenet",
    save_path=None,
    normalize=False,
    no_finetune=False,
    get_features=True
):
    # NOTE: we can load imagenet weights here because we will overwrite them with our fine-tuned weights right after
    model = create_model_rs(
        arch=arch,
        encoder_name="resnet34",
        pretrain="aid",
        in_channels=3,
        classes=3,
        get_features=get_features
    )
    model = model.cuda(0)

    # load model weights
    if not no_finetune:
        model.load_state_dict(torch.load(weights))

    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet34", pretrained="imagenet")
    preprocessing = get_preprocessing(pretraining=pretraining)

    # crop the image to be predicted to a size that is divisible by the patch size used
    if im_size == 480:
        preprocessed = preprocess_prediction(
            img, model_preprocessing=preprocessing, normalize=normalize, preprocessing_fn=preprocessing_fn
        )
        _, features = model.forward(preprocessed, return_features=True)
    else:
        raise NotImplementedError("Only 480x480 images are supported for now.")
    return features

def calculate_mpf(dir):

    masks = np.load(dir)

    fractions = np.zeros((masks.shape[0], 3))

    assert(masks.shape == (masks.shape[0], 480, 480))

    for idx, im in enumerate(masks):
        pond = np.sum(im == 0)
        sea_ice = np.sum(im == 1)
        ocean = np.sum(im == 2)
        mpf = pond / (sea_ice + pond)
        ocf = ocean / (sea_ice + pond + ocean)
        sif = (sea_ice + pond) / (ocean + pond + sea_ice)
        fractions[idx][0] = mpf
        fractions[idx][1] = ocf
        fractions[idx][2] = sif

    return fractions

def filter_and_calculate_mpf(mask_dir, probabilities_dir, threshold):
    masks = np.load(mask_dir)
    probabilities = np.load(probabilities_dir)

    # probabilities come in shape (n, 3, 480, 480), we want to have the maximum probability for each pixel
    # as this corresponds to the probability of the assigned class
    probabilities = probabilities.max(axis=1)

    fractions = np.zeros((masks.shape[0], 3))
    mean_probabilities = np.zeros(masks.shape[0])

    for idx, im in enumerate(masks):
        # mean probability of all pixels in image
        mean_probabilities[idx] = np.mean(probabilities[idx])
        prob_mask = np.where(probabilities[idx] < threshold)
        im[prob_mask] = 3
        pond = np.sum(im == 0)
        sea_ice = np.sum(im == 1)
        ocean = np.sum(im == 2)
        mpf = pond / (sea_ice + pond)
        ocf = ocean / (sea_ice + pond + ocean)
        sif = (sea_ice + pond) / (ocean + pond + sea_ice)
        fractions[idx][0] = mpf
        fractions[idx][1] = ocf
        fractions[idx][2] = sif

    return fractions, mean_probabilities
