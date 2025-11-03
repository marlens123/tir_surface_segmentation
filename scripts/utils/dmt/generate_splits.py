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

import random
import os

# Configurations (! Change the filenames in float, e.g. 29.75)

# sets_dir = os.path.join(base_voc, "ImageSets/Segmentation")
# whole_train_set = "trainaug.txt"
# splits = [2, 4, 8, 20, 105.82]

sets_dir = os.path.join("data/dmt/sets_dir/")
whole_unlabeled_train_set = "pseudo_file_names.txt"
# careful: we change logice here. The splits now indicate the unlabeled data fraction, e.g. 2 means half of the unlabeled data,
# 4 means a quarter of the unlabeled data, etc.
splits = [2, 4, 8, 20, 29.75]

random.seed(7777)

# Open original file
with open(os.path.join(sets_dir, whole_unlabeled_train_set), "r") as f:
    file_names = f.readlines()
original_unlabeled_train_size = len(file_names)
print("Original training set size: " + str(original_unlabeled_train_size))

# ! Check for line EOF
if '\n' not in file_names[original_unlabeled_train_size - 1]:
    file_names[original_unlabeled_train_size - 1] += "\n"

# 3 random splits
# So as to guarantee smaller sets are included in bigger sets
for i in range(3):
    random.shuffle(file_names)

    # Semi-supervised splits
    for split in splits:
        split_index = int(original_unlabeled_train_size / split)  # Floor
        if split_index % 8 == 1:  # For usual batch-size (avoid BN problems)
            split_index += 1
        with open(os.path.join(sets_dir, str(split) + "_unlabeled_" + str(i) + ".txt"), "w") as f:
            f.writelines(file_names[0 : split_index])

    # Whole set (fully-supervised & fully-unsupervised), to be consistent in naming
    with open(os.path.join(sets_dir, "1_unlabeled_" + str(i) + ".txt"), "w") as f:
        f.writelines(file_names)