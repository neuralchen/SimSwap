#!/bin/bash
conda create -n torch-gpu python=3.9
conda activate torch-gpu
conda install pytorch torchvision torchaudio -c pytorch-nightly
conda install -c conda-forge jupyter jupyterlab
python3 -m pip install numpy
python3 -m pip install torch
python3 -m pip install image
python3 -m pip install timm
python3 -m pip install PlL
python3 m1_test.py

