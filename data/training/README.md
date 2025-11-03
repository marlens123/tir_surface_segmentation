## Description

This folder contains our manually annotated data that can be used for training and testing the segmentation models.

- `all_images.npy` and `all_masks.npy` contain the entire set of 21 labeled images from five helicopter flights.
- `train_images.npy` and `train_masks.npy` contain 11 manually selected training images of the labeled set for crossfold validation.
- `test_images.npy` and `test_masks.npy` contain the 10 remaining images as test set for crossfold validation.
