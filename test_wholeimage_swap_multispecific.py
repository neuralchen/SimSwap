'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:22
Description: 
'''

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
import torch.nn as nn
from util.norm import SpecificNorm
import glob
from parsing_model.model import BiSeNet

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def _toarctensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

if __name__ == '__main__':
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    multisepcific_dir = opt.multisepcific_dir

    torch.nn.Module.dump_patches = True

    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'

    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    model = create_model(opt)
    model.eval()
    mse = torch.nn.MSELoss().cuda()

    spNorm =SpecificNorm()


    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode = mode)

    with torch.no_grad():
        # The specific person to be swapped(source)

        source_specific_id_nonorm_list = []
        source_path = os.path.join(multisepcific_dir,'SRC_*')
        source_specific_images_path = sorted(glob.glob(source_path))

        for source_specific_image_path in source_specific_images_path:
            specific_person_whole = cv2.imread(source_specific_image_path)
            specific_person_align_crop, _ = app.get(specific_person_whole,crop_size)
            specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB)) 
            specific_person = transformer_Arcface(specific_person_align_crop_pil)
            specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
            # convert numpy to tensor
            specific_person = specific_person.cuda()
            #create latent id
            specific_person_downsample = F.interpolate(specific_person, size=(112,112))
            specific_person_id_nonorm = model.netArc(specific_person_downsample)
            source_specific_id_nonorm_list.append(specific_person_id_nonorm.clone())


        # The person who provides id information (list)
        target_id_norm_list = []
        target_path = os.path.join(multisepcific_dir,'DST_*')
        target_images_path = sorted(glob.glob(target_path))

        for target_image_path in target_images_path:
            img_a_whole = cv2.imread(target_image_path)
            img_a_align_crop, _ = app.get(img_a_whole,crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            # convert numpy to tensor
            img_id = img_id.cuda()
            #create latent id
            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
            target_id_norm_list.append(latend_id.clone())

        assert len(target_id_norm_list) == len(source_specific_id_nonorm_list), "The number of images in source and target directory must be same !!!"

        ############## Forward Pass ######################

        pic_b = opt.pic_b_path
        img_b_whole = cv2.imread(pic_b)

        img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)
        # detect_results = None
        swap_result_list = []

        id_compare_values = [] 
        b_align_crop_tenor_list = []
        for b_align_crop in img_b_align_crop_list:

            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

            b_align_crop_tenor_arcnorm = spNorm(b_align_crop_tenor)
            b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112,112))
            b_align_crop_id_nonorm = model.netArc(b_align_crop_tenor_arcnorm_downsample)

            id_compare_values.append([])
            for source_specific_id_nonorm_tmp in source_specific_id_nonorm_list:
                id_compare_values[-1].append(mse(b_align_crop_id_nonorm,source_specific_id_nonorm_tmp).detach().cpu().numpy())
            b_align_crop_tenor_list.append(b_align_crop_tenor)

        id_compare_values_array = np.array(id_compare_values).transpose(1,0)
        min_indexs = np.argmin(id_compare_values_array,axis=0)
        min_value = np.min(id_compare_values_array,axis=0)

        swap_result_list = [] 
        swap_result_matrix_list = []
        swap_result_ori_pic_list = []

        for tmp_index, min_index in enumerate(min_indexs):
            if min_value[tmp_index] < opt.id_thres:
                swap_result = model(None, b_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
                swap_result_list.append(swap_result)
                swap_result_matrix_list.append(b_mat_list[tmp_index])
                swap_result_ori_pic_list.append(b_align_crop_tenor_list[tmp_index])
            else:
                pass

        if len(swap_result_list) !=0:

            if opt.use_mask:
                n_classes = 19
                net = BiSeNet(n_classes=n_classes)
                net.cuda()
                save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
                net.load_state_dict(torch.load(save_pth))
                net.eval()
            else:
                net =None
        
            reverse2wholeimage(swap_result_ori_pic_list, swap_result_list, swap_result_matrix_list, crop_size, img_b_whole, logoclass,\
                os.path.join(opt.output_path, 'result_whole_swap_multispecific.jpg'), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)

            print(' ')

            print('************ Done ! ************')
        
        else:
            print('The people you specified are not found on the picture: {}'.format(pic_b))
