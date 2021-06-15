# SimSwap: An Efficient Framework For High Fidelity Face Swapping
## Proceedings of the 28th ACM International Conference on Multimedia
**The official repository with Pytorch**

Currently, only the test code is available, and training scripts are coming soon
[![simswaplogo](/doc/img/logo1.png)](https://github.com/neuralchen/SimSwap)


[[Arxiv paper]](https://arxiv.org/pdf/2106.06340v1.pdf)

[[ACM DOI paper]](https://dl.acm.org/doi/10.1145/3394171.3413630)

[[Google Drive Paper link]](https://drive.google.com/file/d/1fcfWOGt1mkBo7F0gXVKitf8GJMAXQxZD/view?usp=sharing)


[[Baidu Drive Paper link]](https://pan.baidu.com/s/1-TKFuycRNUKut8hn4IimvA) Password: ```ummt```



## Results
![Results1](/doc/img/results1.PNG)

![Results2](/doc/img/total.PNG)

## Video
<img src="./doc/img/video.webp"/>

**High-quality videos can be found in the link below:**

[[Google Drive link for video 1]](https://drive.google.com/file/d/1hdne7Gw39d34zt3w1NYV3Ln5cT8PfCNm/view?usp=sharing)

[[Google Drive link for video 2]](https://drive.google.com/file/d/1oftHAnLmgFis4XURcHTccGSWbWSXYKK1/view?usp=sharing)

[[Baidu Drive link for video]](https://pan.baidu.com/s/1WTS6jm2TY17bYJurw57LUg ) Password: ```b26n```

[[Online Video]](https://www.bilibili.com/video/BV12v411p7j5/)


## Dependencies
- python3.6+
- pytorch1.5+
- torchvision
- opencv
- pillow
- numpy


## Usage
### To test the pretrained model
```
python test_one_image.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path crop_224/6.jpg --pic_b_path crop_224/ds.jpg --output_path output/
```

--name refers to the SimSwap training logs name.

## Pretrained model

### Usage
There are two archive files in the drive: **checkpoints.zip** and **arcface_checkpoint.tar**

- **Copy the arcface_checkpoint.tar into ./arcface_model**
- **Unzip checkpoints.zip, place it in the root dir ./**

[[Google Drive]](https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R?usp=sharing)

[[Baidu Drive]](https://pan.baidu.com/s/1wFV11RVZMHqd-ky4YpLdcA) Password: ```jd2v```


## To cite our paper
```
@inproceedings{DBLP:conf/mm/ChenCNG20,
  author    = {Renwang Chen and
               Xuanhong Chen and
               Bingbing Ni and
               Yanhao Ge},
  title     = {SimSwap: An Efficient Framework For High Fidelity Face Swapping},
  booktitle = {{MM} '20: The 28th {ACM} International Conference on Multimedia},
  pages     = {2003--2011},
  publisher = {{ACM}},
  year      = {2020},
  url       = {https://doi.org/10.1145/3394171.3413630},
  doi       = {10.1145/3394171.3413630},
  timestamp = {Thu, 15 Oct 2020 16:32:08 +0200},
  biburl    = {https://dblp.org/rec/conf/mm/ChenCNG20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Related Projects

**Please visit our another ACMMM2020 high-quality style transfer project**

[![logo](./doc/img/logo.png)](https://github.com/neuralchen/ASMAGAN)

[![title](/doc/img/title.png)](https://github.com/neuralchen/ASMAGAN)

Learn about our other projects 
[[RainNet]](https://neuralchen.github.io/RainNet);

[[Sketch Generation]](https://github.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale);

[[CooGAN]](https://github.com/neuralchen/CooGAN);

[[Knowledge Style Transfer]](https://github.com/AceSix/Knowledge_Transfer);

[[SimSwap]](https://github.com/neuralchen/SimSwap);

[[ASMA-GAN]](https://github.com/neuralchen/ASMAGAN);

[[SNGAN-Projection-pytorch]](https://github.com/neuralchen/SNGAN_Projection)

[[Pretrained_VGG19]](https://github.com/neuralchen/Pretrained_VGG19).