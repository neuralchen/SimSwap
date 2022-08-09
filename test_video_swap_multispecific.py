
import cv2
import torch
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap_multispecific import video_swap
import os
import glob

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])


if __name__ == '__main__':
    opt = TestOptions().parse()
    pic_specific = opt.pic_specific_path
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
    model = create_model(opt)
    model.eval()


    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    # The specific person to be swapped(source)

    source_specific_id_nonorm_list = []
    source_path = os.path.join(multisepcific_dir,'SRC_*')
    source_specific_images_path = sorted(glob.glob(source_path))
    with torch.no_grad():
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



        video_swap(opt.video_path, target_id_norm_list,source_specific_id_nonorm_list, opt.id_thres, \
            model, app, opt.output_path,temp_results_dir=opt.temp_path,no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size)

