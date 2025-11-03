from sklearn.utils import class_weight
import numpy as np
import torch
import random
import torch.nn.functional as F
from models.AutoSAM.loss_functions.dice_loss import soft_dice_per_batch_2
import cv2

def compute_class_weights(train_masks):

    if isinstance(train_masks, str):
        train_masks = np.load(train_masks)

    masks_resh = train_masks.reshape(-1, 1)
    masks_resh_list = masks_resh.flatten().tolist()
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(masks_resh), y=masks_resh_list
    )
    return class_weights

def compute_class_frequencies(train_masks):
    
    if isinstance(train_masks, str):
        train_masks = np.load(train_masks)

    total_pixels = train_masks.flatten().shape[0]
    class_frequencies = []
    for cls in range(3):  # assuming classes are 0, 1, 2
        class_count = np.sum(train_masks == cls)
        class_frequency = class_count / total_pixels
        class_frequencies.append(class_frequency)

    return class_frequencies

def compute_pixel_distance_to_edge(train_masks, teta=3.0):
    # compute the pixel-wise distance to the edge of the object
    if isinstance(train_masks, str):
        train_masks = np.load(train_masks)

    all_distances = []
    for i in range(train_masks.shape[0]):
        mask = train_masks[i]
        edges_class_1 = cv2.Canny((mask == 1).astype(np.uint8) * 255, 100, 200)
        edges_class_2 = cv2.Canny((mask == 2).astype(np.uint8) * 255, 100, 200)
        edges_class_3 = cv2.Canny((mask == 3).astype(np.uint8) * 255, 100, 200)
        # combine all edges into one image
        edges = edges_class_1 | edges_class_2 | edges_class_3
        edges[edges > 0] = 1

        dist_transform = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 3)
        # exponentially decay the distance transform
        dist_transform = np.exp(-dist_transform / teta)
        # invert the distance transform
        dist_transform = 1 - dist_transform

        # normalize to 0-1
        dist_transform = dist_transform / np.max(dist_transform)
        all_distances.append(dist_transform)

    all_distances = np.array(all_distances)
    return all_distances


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("Setting seed for GPU")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False