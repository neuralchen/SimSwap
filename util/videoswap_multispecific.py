import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
import torch.nn.functional as F
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, target_id_norm_list,source_specific_id_nonorm_list,id_thres, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    video_forcheck = VideoFileClip(video_path)
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if  os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)

    spNorm =SpecificNorm()
    mse = torch.nn.MSELoss().cuda()

    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    # while ret:
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            detect_results = detect_model.get(frame,crop_size)

            if detect_results is not None:
                # print(frame_index)
                if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                frame_align_crop_list = detect_results[0]
                frame_mat_list = detect_results[1]

                id_compare_values = [] 
                frame_align_crop_tenor_list = []
                for frame_align_crop in frame_align_crop_list:

                    # BGR TO RGB
                    # frame_align_crop_RGB = frame_align_crop[...,::-1]

                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                    frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
                    frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, scale_factor=0.5)
                    frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)
                    id_compare_values.append([])
                    for source_specific_id_nonorm_tmp in source_specific_id_nonorm_list:
                        id_compare_values[-1].append(mse(frame_align_crop_crop_id_nonorm,source_specific_id_nonorm_tmp).detach().cpu().numpy())
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                id_compare_values_array = np.array(id_compare_values).transpose(1,0)
                min_indexs = np.argmin(id_compare_values_array,axis=0)
                min_value = np.min(id_compare_values_array,axis=0)

                swap_result_list = [] 
                swap_result_matrix_list = []
                swap_result_ori_pic_list = []
                for tmp_index, min_index in enumerate(min_indexs):
                    if min_value[tmp_index] < id_thres:
                        swap_result = swap_model(None, frame_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
                        swap_result_list.append(swap_result)
                        swap_result_matrix_list.append(frame_mat_list[tmp_index])
                        swap_result_ori_pic_list.append(frame_align_crop_tenor_list[tmp_index])
                    else:
                        pass



                if len(swap_result_list) !=0:
                    
                    reverse2wholeimage(swap_result_ori_pic_list,swap_result_list, swap_result_matrix_list, crop_size, frame, logoclass,\
                        os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask, norm = spNorm)
                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    if not no_simswaplogo:
                        frame = logoclass.apply_frames(frame)
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
        else:
            break

    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)


    clips.write_videofile(save_path,audio_codec='aac')

