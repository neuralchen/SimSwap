#!/bin/bash

conda init --all
conda create --name tensorflow_m1 python==3.9
conda activate tensorflow_m1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
python3 -m pip install insightface==0.2.1 moviepy
python3 -m pip install googledrivedownloader
python3 -m pip install imageio==2.4.1
python3 -m pip install numpy
python3 -m pip install torch
python3 -m pip install image
python3 -m pip install timm
python3 -m pip install PlL
python3 -m pip install wrapt==1.12.1
python3 -m pip install opt_einsum
python3 -m pip install flatbuffers==1.12.0
python3 -m pip install google-pasta==0.2
python3 -m pip install h5py
python3 -m pip install keras-nightly==2.10.0.dev2022071007
python3 -m pip install keras-preprocessing==1.1.2
python3 -m pip install termcolor==1.1.0
python3 -m pip install absl-py==0.10
python3 -m pip install gast==0.4.0
python3 -m pip install grpcio
python3 -m pip install typing-extensions==3.7.4
python3 -m pip install astunparse==1.6.3
conda install -c apple tensorflow-deps
python3 -m pip install tensorflow-macos
python3 -m pip install tensorflow-metal==0.1.2
python3 -m pip install tensorflow-estimator<2.6.0,>=2.5.0rc0
python3 -m pip install tensorboard<1.13.0,>=1.12.0

python3 m1_tf_test.py

conda deactivate

exit 0
