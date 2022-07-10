#!/bin/bash
export HOMEBREW_BREW_GIT_REMOTE="https://github.com/Homebrew/brew"  # put your Git mirror of Homebrew/brew here
export HOMEBREW_CORE_GIT_REMOTE="https://github.com/Homebrew/homebrew-core"  # put your Git mirror of Homebrew/homebrew-core here
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"
brew update --force --quiet
chmod -R go-w "$(brew --prefix)/share/zsh"
export OPENBLAS=$(/opt/homebrew/bin/brew --prefix openblas)
export CFLAGS="-falign-functions=8 ${CFLAGS}"

conda init --all
conda create --name tensorflow_m1 python==3.9
conda activate tensorflow_m1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
python3 -m venv ~/tensorflow-metal
source ~/tensorflow-metal/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install insightface==0.2.1 moviepy
python3 -m pip install googledrivedownloader
python3 -m pip install imageio==2.4.1
python3 -m pip install Cython
python3 -m pip install --no-use-pep517 numpy==1.21.2
python3 -m pip install torch
python3 -m pip install image
python3 -m pip install timm
python3 -m pip install PlL
python3 -m pip install wrapt
python3 -m pip install opt_einsum
python3 -m pip install flatbuffers
python3 -m pip install google-pasta
python3 -m pip install h5py
python3 -m pip install keras-nightly
python3 -m pip install keras-preprocessing
python3 -m pip install termcolor
python3 -m pip install absl-py
python3 -m pip install gast
python3 -m pip install grpcio
python3 -m pip install typing-extensions
python3 -m pip install astunparse
conda install -c apple tensorflow-deps
python3 -m pip install tensorflow-macos
python3 -m pip install tensorflow-metal
python3 -m pip install tensorflow-estimator
python3 -m pip install tensorboard

python3 m1_tf_test.py

conda deactivate

exit 0
