# SimSwap: An Efficient Framework For High Fidelity Face Swapping
## Proceedings of the 28th ACM International Conference on Multimedia
## The official repository with Pytorch
[[Conference paper]](https://dl.acm.org/doi/10.1145/3394171.3413630)

![Results1](/doc/img/results1.PNG)
![Results2](/doc/img/results2.PNG)

Use python3.5, pytorch1.3.0


Use this command to test the face swapping between two images:

python test_one_image.py --isTrain false  --name people --Arc_path models/BEST_checkpoint.tar --pic_a_path crop_224/mars.jpg --pic_b_path crop_224/ds.jpg --output_path output/

--name refers to the checkpoint name.
