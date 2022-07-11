#!/bin/bash
conda init --all
conda create --name tensorflow_m1 python==3.9
conda activate tensorflow_m1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
python3 -m venv ~/tensorflow-metal
source ~/tensorflow-metal/bin/activate

for i in `seq 1 6`; do
python3 m1_tf_test.py $(( ( RANDOM % 4096 )  + 4096 ))
done

conda deactivate

exit 0
