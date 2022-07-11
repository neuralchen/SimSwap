#!/bin/bash
conda init --all
conda create -n torch-gpu python=3.9
conda activate torch-gpu
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch

for i in `seq 1 6`; do
python3 m1_test.py
done

conda deactivate

exit 0
