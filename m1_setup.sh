#!/bin/bash
export HOMEBREW_BREW_GIT_REMOTE="https://github.com/Homebrew/brew"  # put your Git mirror of Homebrew/brew here
export HOMEBREW_CORE_GIT_REMOTE="https://github.com/Homebrew/homebrew-core"  # put your Git mirror of Homebrew/homebrew-core here
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"
brew update --force --quiet
chmod -R go-w "$(brew --prefix)/share/zsh"
brew install wget
brew install unzip

conda init --all
conda create -n torch-gpu python=3.9
conda activate torch-gpu
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge jupyter jupyterlab
#wget -P ./arcface_model https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar
#wget https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip
#unzip ./checkpoints.zip  -d ./checkpoints
#wget -P ./parsing_model/checkpoint https://github.com/neuralchen/SimSwap/releases/download/1.0/79999_iter.pth
#wget --no-check-certificate "https://sh23tw.dm.files.1drv.com/y4mmGiIkNVigkSwOKDcV3nwMJulRGhbtHdkheehR5TArc52UjudUYNXAEvKCii2O5LAmzGCGK6IfleocxuDeoKxDZkNzDRSt4ZUlEt8GlSOpCXAFEkBwaZimtWGDRbpIGpb_pz9Nq5jATBQpezBS6G_UtspWTkgrXHHxhviV2nWy8APPx134zOZrUIbkSF6xnsqzs3uZ_SEX_m9Rey0ykpx9w" -O antelope.zip
#unzip ./antelope.zip -d ./insightface_func/models/

python3 -m pip install --ignore-installed imageio
python3 -m pip install --ignore-installed numpy
python3 -m pip install --ignore-installed wrapt
python3 -m pip install --ignore-installed flatbuffers
python3 -m pip install --ignore-installed google-pasta
python3 -m pip install --ignore-installed h5py
python3 -m pip install --ignore-installed keras-nightly
python3 -m pip install --ignore-installed keras-preprocessing
python3 -m pip install --ignore-installed tensorflow-estimator
python3 -m pip install --ignore-installed termcolor
python3 -m pip install --ignore-installed absl
python3 -m pip install --ignore-installed gast
python3 -m pip install --ignore-installed grpcio
python3 -m pip install --ignore-installed typing-extentions
python3 -m pip install --ignore-installed astunparse
python3 -m pip install insightface==0.2.1 onnxruntime moviepy
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
python3 -m pip install h5py==3.7.0
python3 -m pip install keras-nightly==2.5.0.dev
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

python3 m1_test.py
for i in `seq 1 6`; do
python3 train.py --name simswap224_test --batchSize 8000000000000  --gpu_ids 0 --dataset crop_224/$i.jpg
done

conda deactivate

exit 0
