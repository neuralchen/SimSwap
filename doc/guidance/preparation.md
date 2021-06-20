
# Preparation

### Installation
**We highly recommand that you use Anaconda for Installation**
```
conda create -n simswap python=3.6
conda activate simswap
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
(option): pip install --ignore-installed imageio
pip install insightface==0.2.1 onnxruntime moviepy
```
- We use the face detection and alignment methods from **[insightface](https://github.com/deepinsight/insightface)** for image preprocessing. Please download the relative files and unzip them to ./insightface_func/models from [this link](https://onedrive.live.com/?authkey=%21ADJ0aAOSsc90neY&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215837&parId=4A83B6B633B029CC%215834&action=locate).
- The pytorch and cuda versions above are most recommanded. They may vary.
- Using insightface with different versions is not recommanded. Please use this specific version.
- These settings are tested valid on both Windows and Ununtu.

### Pretrained model
There are two archive files in the drive: **checkpoints.zip** and **arcface_checkpoint.tar**

- **Copy the arcface_checkpoint.tar into ./arcface_model**
- **Unzip checkpoints.zip, place it in the root dir ./**

[[Google Drive]](https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R?usp=sharing)
[[Baidu Drive]](https://pan.baidu.com/s/1wFV11RVZMHqd-ky4YpLdcA) Password: ```jd2v```

### Note
We expect users to have GPU with at least 8G memory. For those who do not, we provide [[Colab Notebook implementation]](https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/SimSwap%20colab.ipynb).
