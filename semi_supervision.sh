#!/bin/bash

train_set="1"
sid="0"
exp_name="dmt-melt-ponds-trail1"
ep="4"
lr="0.0005"
c1="5"
c2="5"
i1="5"
i2="5"
cid="default"
phases=("1" "2" "3" "4" "5")
rates=("0.2" "0.4" "0.6" "0.8" "1.0")

echo initial training
python -m scripts.run_dmt --exp-name=${exp_name}__p0--a --val-num-steps=220 --state=2 --epochs=150 --train-set=${train_set} --sets_id=${sid} --aid --mixed-precision --batch-size-labeled=4 --batch-size-pseudo=0 --seed=42 --lr=${lr} --pref=${cid}
python -m scripts.run_dmt --exp-name=${exp_name}__p0--i --val-num-steps=220 --state=2 --epochs=150 --train-set=${train_set} --sets_id=${sid} --mixed-precision --batch-size-labeled=4 --batch-size-pseudo=0 --seed=42 --lr=${lr} --pref=${cid}

echo dmt
for i in ${!rates[@]}; do
  echo ${phases[$i]}--${rates[$i]}
  
  echo labeling
  python -m scripts.run_dmt --labeling --state=1 --train-set=${train_set} --sets_id=${sid} --continue-from=dmt_checkpoints_${cid}/${exp_name}__p${i}--i.pt --mixed-precision --batch-size-labeled=1 --label_ratio=${rates[$i]} --pref=${cid}

  echo training
  python -m scripts.run_dmt --state=1 --exp-name=${exp_name}__p${phases[$i]}--a --train-set=${train_set} --sets_id=${sid} --continue-from=dmt_checkpoints_${cid}/${exp_name}__p${i}--a.pt --aid --mixed-precision --epochs=${ep} --gamma1=${c1} --gamma2=${c2} --lr=${lr} --seed=1 --pref=${cid}

  echo labeling
  python -m scripts.run_dmt --labeling --state=1 --train-set=${train_set} --sets_id=${sid} --continue-from=dmt_checkpoints_${cid}/${exp_name}__p${i}--a.pt --aid --mixed-precision --batch-size-labeled=1 --label_ratio=${rates[$i]} --pref=${cid}

  echo training
  python -m scripts.run_dmt --state=1 --exp-name=${exp_name}__p${phases[$i]}--i --train-set=${train_set} --sets_id=${sid} --continue-from=dmt_checkpoints_${cid}/${exp_name}__p${i}--i.pt --mixed-precision --epochs=${ep} --gamma1=${i1} --gamma2=${i2} --lr=${lr} --seed=2 --pref=${cid}

done