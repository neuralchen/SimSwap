<!--
 * @FilePath: \SimSwap\docs\guidance\usage.md
 * @Author: AceSix
 * @Date: 2021-06-28 10:01:40
 * @LastEditors: AceSix
 * @LastEditTime: 2021-06-28 10:05:11
 * Copyright (C) 2021 SJTU. All rights reserved.
-->

# Usage

###### Before running, please make sure you have installed the environment and downloaded requested files according to the [preparation guidance](./preparation.md).

### Simple face swapping for already face-aligned images
```
python test_one_image.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path crop_224/6.jpg --pic_b_path crop_224/ds.jpg --output_path output/
```

### Face swapping for video

- Swap only one face within the video(the one with highest confidence by face detection).
```
python test_video_swapsingle.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/Iron_man.jpg --video_path ./demo_file/mutil_people_1080p.mp4 --output_path ./output/mutil_test_swapsingle.mp4 --temp_path ./temp_results
```
- Swap all faces within the video.
```
python test_video_swapmutil.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/Iron_man.jpg --video_path ./demo_file/mutil_people_1080p.mp4 --output_path ./output/mutil_test_swapmutil.mp4 --temp_path ./temp_results
```
- Swap the ***specific*** face within the video.
```
python test_video_swapspecific.py --pic_specific_path ./demo_file/specific1.png --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/Iron_man.jpg --video_path ./demo_file/mutil_people_1080p.mp4 --output_path ./output/mutil_test_specific.mp4 --temp_path ./temp_results 
```
When changing the specified face, you need to give a picture of the person whose face is to be changed. Then assign the picture path to the argument "***--pic_specific_path***". This picture should be a front face and show the entire head and neck, which can help accurately change the face (if you still don’t know how to choose the picture, you can refer to the specific*.png of [./demo_file/](https://github.com/neuralchen/SimSwap/tree/main/demo_file)). It would be better if this picture was taken from the video to be changed.

- Swap ***multi specific*** face with **multi specific id** within the video.
```
python test_video_swap_mutilspecific.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --video_path ./demo_file/mutil_people_1080p.mp4 --output_path ./output/mutil_test_mutilspecific.mp4 --temp_path ./temp_results --mutilsepcific_dir ./demo_file/mutilspecific
```
The folder you assign to ***"--mutilsepcific_dir"*** should be looked like:
```
$Your folder name$
├── source
│   ├── 01.jpg(png)
│   └── 02.jpg(png)
│   └──...
├── target
│   ├── 01.jpg(png)
│   └── 02.jpg(png)
│   └──...
```
The result is that the person corresponding to 01.jpg (png) in the source dir in the video will be replaced with the face of the person corresponding to 01.jpg (png) in the target dir. Then the person corresponding to 02.jpg (png) in source dir will be replaced with the face of 02.jpg (png) in target dir, and so on. Note that when you use your own data and name it, do not remove the **0** in **0**1.jpg(png), etc.


### Face swapping for Arbitrary images

- Swap only one face within one image(the one with highest confidence by face detection). The result would be saved to ./output/result_whole_swapsingle.jpg
```
python test_wholeimage_swapsingle.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/Iron_man.jpg --pic_b_path ./demo_file/mutil_people.jpg --output_path ./output/
```
- Swap all faces within one image. The result would be saved to ./output/result_whole_swapmutil.jpg
```
python test_wholeimage_swapmutil.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/Iron_man.jpg --pic_b_path ./demo_file/mutil_people.jpg --output_path ./output/
```
- Swap **specific** face within one image. The result would be saved to ./output/result_whole_swapspecific.jpg
```
python test_wholeimage_swapspecific.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/Iron_man.jpg --pic_b_path ./demo_file/mutil_people.jpg --output_path ./output/ --pic_specific_path ./demo_file/specific2.png
```
- Swap **multi specific** face with **multi specific id** within one image. The result would be saved to ./output/result_whole_swap_mutilspecific.jpg
```
python test_wholeimage_swap_mutilspecific.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_b_path ./demo_file/mutil_people.jpg --output_path ./output/ --mutilsepcific_dir ./demo_file/mutilspecific
```
### About watermark of simswap logo
The above example command line is to add the simswap logo as the watermark by default. After our discussion, we have added a hyper parameter to control whether to remove watermark.

The usage of removing the watermark is to add an argument: "***--no_simswaplogo***" to the command line, take the command line of "Swap all faces within one image" as an example, the following command line can get the result without watermark:
```
python test_wholeimage_swapmutil.py --no_simswaplogo --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/Iron_man.jpg --pic_b_path ./demo_file/mutil_people.jpg --output_path ./output/
```




Difference between single face swapping and all face swapping are shown below.
<img src="../img/multi_face_comparison.png"/>

### Parameters
|  Parameters   | Function  |
|  :----  | :----  |
| --name  | The SimSwap training logs name |
| --pic_a_path  | Path of image with the target face |
| --pic_b_path  | Path of image with the source face to swap |
| --pic_specific_path  | Path of image with the specific face to be swapped |
|--mutilsepcific_dir  |Path of image folder for multi specific face swapping|
| --video_path  | Path of video with the source face to swap |
| --temp_path  | Path to store intermediate files  |
| --output_path  | Path of directory to store the face swapping result  |
| --no_simswaplogo  |The hyper parameter to control whether to remove watermark |

### Note
We expect users to have GPU with at least 8G memory. For those who do not, we will provide Colab Notebook implementation in the future.
