"""
This script creates a pseudo-labeled dataset for DMT training by sampling from the available flights while avoiding certain forbidden indices.
Forbidden are all indices that are already labeled, as well as 4 indices before and after each labeled index.
Also, the sampling is done according to the ratio of the number of files in each flight, so that the final dataset is balanced.
No overlapping samples are allowed, i.e., no two samples can be within 4 indices of each other.
"""

from pathlib import Path
import os
import numpy as np
import cv2

# forbid all indeces that are in the labeled set
flight07_forbidden_indeces = [8]
flight09_forbidden_indeces = [1024, 1104, 2380, 2416, 2424, 2452, 2468, 2476, 2708, 3700, 3884, 4568]
flight10_forbidden_indeces = []
flight11_forbidden_indeces = [8, 40]
flight16_forbidden_indeces = [100, 200, 256, 64, 260, 552]
flight17_forbidden_indeces = []

complete_flight07_forbidden_indeces = flight07_forbidden_indeces.copy()
complete_flight09_forbidden_indeces = flight09_forbidden_indeces.copy()
complete_flight10_forbidden_indeces = flight10_forbidden_indeces.copy()
complete_flight11_forbidden_indeces = flight11_forbidden_indeces.copy()
complete_flight16_forbidden_indeces = flight16_forbidden_indeces.copy()
complete_flight17_forbidden_indeces = flight17_forbidden_indeces.copy()

# also forbid 4 indeces before and after each labeled index for each respective flight
forbidden_range = 4

for idx in flight07_forbidden_indeces:
    # append all indeces in the range to the forbidden indeces list if not already present
    for i in range(idx - forbidden_range, idx + forbidden_range + 1):
        complete_flight07_forbidden_indeces.append(i)
    # delete duplicates
    complete_flight07_forbidden_indeces = list(set(complete_flight07_forbidden_indeces))

for idx in flight09_forbidden_indeces:
    for i in range(idx - forbidden_range, idx + forbidden_range + 1):
        if i not in complete_flight09_forbidden_indeces:
            complete_flight09_forbidden_indeces.append(i)
    complete_flight09_forbidden_indeces = list(set(complete_flight09_forbidden_indeces))

for idx in flight10_forbidden_indeces:
    for i in range(idx - forbidden_range, idx + forbidden_range + 1):
        if i not in complete_flight10_forbidden_indeces:
            complete_flight10_forbidden_indeces.append(i)
    complete_flight10_forbidden_indeces = list(set(complete_flight10_forbidden_indeces))

for idx in flight11_forbidden_indeces:
    for i in range(idx - forbidden_range, idx + forbidden_range + 1):
        if i not in complete_flight11_forbidden_indeces:
            complete_flight11_forbidden_indeces.append(i)
    complete_flight11_forbidden_indeces = list(set(complete_flight11_forbidden_indeces))

for idx in flight16_forbidden_indeces:
    for i in range(idx - forbidden_range, idx + forbidden_range + 1):
        if i not in complete_flight16_forbidden_indeces:
            complete_flight16_forbidden_indeces.append(i)
    complete_flight16_forbidden_indeces = list(set(complete_flight16_forbidden_indeces))

for idx in flight17_forbidden_indeces:
    for i in range(idx - forbidden_range, idx + forbidden_range + 1):
        if i not in complete_flight17_forbidden_indeces:
            complete_flight17_forbidden_indeces.append(i)
    complete_flight17_forbidden_indeces = list(set(complete_flight17_forbidden_indeces))

preprocessed_path_flight07 = Path("data/prediction/preprocessed/IRdata_ATWAICE_processed_220717_122355.nc/")
preprocessed_path_flight09 = Path("data/prediction/preprocessed/IRdata_ATWAICE_processed_220718_142920.nc/")
preprocessed_path_flight10 = Path("data/prediction/preprocessed/IRdata_ATWAICE_processed_220719_104906.nc/")
preprocessed_path_flight11 = Path("data/prediction/preprocessed/IRdata_ATWAICE_processed_220719_112046.nc/")
preprocessed_path_flight16 = Path("data/prediction/preprocessed/IRdata_ATWAICE_processed_220730_111439.nc/")
preprocessed_path_flight17 = Path("data/prediction/preprocessed/IRdata_ATWAICE_processed_220808_084908.nc/")

# create ratios for each flight, incorporating the number of files, all flight should sum to 1
num_files_flight11 = len(os.listdir(preprocessed_path_flight11))
num_files_flight07 = len(os.listdir(preprocessed_path_flight07))
num_files_flight10 = len(os.listdir(preprocessed_path_flight10))
num_files_flight09 = len(os.listdir(preprocessed_path_flight09))
num_files_flight16 = len(os.listdir(preprocessed_path_flight16))

# exclude flight 17 because it's too different from the others
#num_files_flight17 = len(os.listdir(preprocessed_path_flight17))

total_files = num_files_flight11 + num_files_flight07 + num_files_flight10 + num_files_flight09 + num_files_flight16

ratio_flight11 = num_files_flight11 / total_files
ratio_flight07 = num_files_flight07 / total_files
ratio_flight10 = num_files_flight10 / total_files
ratio_flight09 = num_files_flight09 / total_files
ratio_flight16 = num_files_flight16 / total_files
#ratio_flight17 = num_files_flight17 / total_files

# create a csv file with all the forbidden indeces and ratios stored per flight
with open("data/dmt/forbidden_indeces_and_ratios_no_flight17.csv", "w") as f:
    f.write("flight,forbidden_indeces,ratio\n")
    f.write(f"flight07,{complete_flight07_forbidden_indeces},{ratio_flight07}\n")
    f.write(f"flight09,{complete_flight09_forbidden_indeces},{ratio_flight09}\n")
    f.write(f"flight10,{complete_flight10_forbidden_indeces},{ratio_flight10}\n")
    f.write(f"flight11,{complete_flight11_forbidden_indeces},{ratio_flight11}\n")
    f.write(f"flight16,{complete_flight16_forbidden_indeces},{ratio_flight16}\n")
    #f.write(f"flight17,{complete_flight17_forbidden_indeces},{ratio_flight17}\n")

# now, sample from each flight according to the ratio, but only from the allowed indeces
all_images = []
all_masks = []
all_flight_names = []

for i, (flight_path, forbidden_indeces) in enumerate(zip([preprocessed_path_flight07, preprocessed_path_flight09, preprocessed_path_flight10, preprocessed_path_flight11, preprocessed_path_flight16], 
                                                         [complete_flight07_forbidden_indeces, complete_flight09_forbidden_indeces, complete_flight10_forbidden_indeces, complete_flight11_forbidden_indeces, complete_flight16_forbidden_indeces])):
    flight_name = flight_path.stem
    files = sorted(os.listdir(flight_path))
    allowed_files = [f for j, f in enumerate(files) if j not in forbidden_indeces]
    
    if i == 0:
        num_samples = int(ratio_flight07 * 1000)
    elif i == 1:
        num_samples = int(ratio_flight09 * 1000)
    elif i == 2:
        num_samples = int(ratio_flight10 * 1000)
    elif i == 3:
        num_samples = int(ratio_flight11 * 1000)
    elif i == 4:
        num_samples = int(ratio_flight16 * 1000)
    #elif i == 5:
    #    num_samples = int(ratio_flight17 * 1000)

    # sample from all allowed files without replacement, with a step size of forbidden_range to ensure no two sampled files are within forbidden_range of each other
    if num_samples > len(allowed_files):
        num_samples = len(allowed_files)
    step_size = max(1, len(allowed_files) // num_samples)
    sampled_files = allowed_files[::step_size][:num_samples]

    for file in sampled_files:
        img = cv2.imread(os.path.join(flight_path, file), cv2.IMREAD_GRAYSCALE)
        all_images.append(img)
        # create a dummy mask of zeros
        mask = np.zeros_like(img)
        all_masks.append(mask)
        all_flight_names.append(os.path.join(flight_path, file).split("preprocessed/")[-1])

all_images = np.array(all_images)
all_masks = np.array(all_masks)

# write all flight names to a txt file
with open("data/dmt/sets_dir/pseudo_file_names.txt", "w") as f:
    for name in all_flight_names:
        f.write(name + "\n")

# save images and masks as numpy arrays
np.save("data/dmt/images/pseudo_images.npy", all_images)
#np.save("data/dmt/train_masks_pseudo/masks.npy", all_masks)