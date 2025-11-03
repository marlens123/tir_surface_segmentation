# Copyright 2023 Xinrong Hu et al. https://github.com/xhu248/AutoSAM

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/sam/lib

source activate pytorch

# model_path=./save/simclr_ACDC/b_48_f0_model.pth
fold=0
gpu=2
tr_size=1
model_type=vit_b
dataset=ACDC
output_dir=sam_${model_type}_seg_${dataset}_f${fold}_tr_${tr_size}_dec
python scripts/main_autosam_seg.py --src_dir ../DATA/${dataset} \
--data_dir ../DATA/${dataset}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ${dataset} --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4



