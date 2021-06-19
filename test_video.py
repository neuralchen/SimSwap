
import cv2
import torch
import fractions
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

opt = TestOptions().parse()

start_epoch, epoch_iter = 1, 0

torch.nn.Module.dump_patches = True
model = create_model(opt)
model.eval()

def img_b_atte(img_b):
    img_b = transformer(img_b)
    img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])
    img_att = img_att.cuda()
    return img_att

def swap(img_id,img_att,latend_id):
    img_fake = model(img_id, img_att, latend_id, latend_id, True)
    for i in range(img_id.shape[0]):
        if i == 0:
            row1 = img_id[i]
            row2 = img_att[i]
            row3 = img_fake[i]
        else:
            row1 = torch.cat([row1, img_id[i]], dim=2)
            row2 = torch.cat([row2, img_att[i]], dim=2)
            row3 = torch.cat([row3, img_fake[i]], dim=2)
    
    full = row3.detach()
    full = full.permute(1, 2, 0)
    output = full.to('cpu')
    output = np.array(output)
    output = output[..., ::-1]
    output = output*255
    output=output.astype(np.uint8)
    return output

pic_a = opt.pic_a_path
img_a=cv2.imread(pic_a)
img_a=cv2.cvtColor(img_a,cv2.COLOR_BGR2RGB)
h,w,_=img_a.shape
if w!=224 or h!=224:
    img_a=cv2.resize(img_a,(224,224))
img_a = transformer_Arcface(img_a)
img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
img_id=img_id.cuda()

#create latent id
img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
latend_id = model.netArc(img_id_downsample)
latend_id = latend_id.detach().to('cpu')
latend_id = latend_id/np.linalg.norm(latend_id)
latend_id = latend_id.to('cuda')

cap=cv2.VideoCapture(opt.video_path)

while cap.isOpened():
    _,img_b=cap.read()
    if img_b is None:
        break
    h,w,_=img_b.shape
    if w!=224 or h!=224:
        img_b=cv2.resize(img_b,(224,224))
    img_b=cv2.cvtColor(img_b,cv2.COLOR_BGR2RGB)
    img_att=img_b_atte(img_b)
    img_fake=swap(img_id,img_att,latend_id)
    cv2.imshow("swap",img_fake)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
