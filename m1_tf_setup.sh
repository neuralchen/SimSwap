#!/bin/bash

conda init --all
conda create --name tensorflow_m1 python==3.9
conda activate tensorflow_m1
python3 -m pip uninstall imageio
python3 -m pip uninstall numpy
python3 -m pip uninstall wrapt
python3 -m pip uninstall flatbuffers
python3 -m pip uninstall google-pasta
python3 -m pip uninstall h5py
python3 -m pip uninstall keras-nightly
python3 -m pip uninstall keras-preprocessing
python3 -m pip uninstall tensorflow-estimator
python3 -m pip uninstall termcolor
python3 -m pip uninstall absl
python3 -m pip uninstall gast
python3 -m pip uninstall grpcio
python3 -m pip uninstall typing-extentions
python3 -m pip uninstall astunparse
python3 -m pip install onnxruntime-gpu
python3 -m pip install insightface==0.2.1 onnxruntime moviepy
python3 -m pip install googledrivedownloader
python3 -m pip install imageio==2.4.1
python3 -m pip install numpy==1.19.2
python3 -m pip install torch
python3 -m pip install image
python3 -m pip install timm
python3 -m pip install PlL
python3 -m pip install wrapt==1.12.1
python3 -m pip install opt_einsum
python3 -m pip install flatbuffers==1.12.0
python3 -m pip install google-pasta==0.2
python3 -m pip install h5py==3.1.0
python3 -m pip install keras-nightly==2.5.0.dev2021032900
python3 -m pip install keras-preprocessing==1.1.2
python3 -m pip install tensorflow-estimator<2.6.0,>=2.5.0rc0
python3 -m pip install termcolor==1.1.0
python3 -m pip install absl-py==0.10
python3 -m pip install gast==0.4.0
python3 -m pip install grpcio==1.34.0
python3 -m pip install typing-extensions==3.7.4
python3 -m pip install astunparse==1.6.3
conda install -c apple tensorflow-deps==2.5.0
python3 -m pip install tensorflow-macos==2.5.0
python3 -m pip install tensorflow-macos==2.5.0 --no-dependencies
python3 -m pip install tensorflow-metal==0.1.2

python3 m1_tf_test.py

conda deactivate

exit 0