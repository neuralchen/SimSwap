import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from torchvision import transforms
from PIL import Image
import cv2

class FaceDataSet(Dataset):
    def __init__(self, dataset_path, batch_size):
        super(FaceDataSet, self).__init__()



        '''picture_dir_list = []
        for i in range(self.people_num):
            picture_dir_list.append('/data/home/renwangchen/vgg_align_224/'+self.people_list[i])

        self.people_pic_list = []
        for i in range(self.people_num):
            pic_list = os.listdir(picture_dir_list[i])
            person_pic_list = []
            for j in range(len(pic_list)):
                pic_dir = os.path.join(picture_dir_list[i], pic_list[j])
                person_pic_list.append(pic_dir)
            self.people_pic_list.append(person_pic_list)'''

        pic_dir = '/data/home/renwangchen/CelebA_224/'
        latent_dir = '/data/home/renwangchen/CelebA_latent/'

        tmp_list = os.listdir(pic_dir)
        self.pic_list = []
        self.latent_list = []
        for i in range(len(tmp_list)):
            self.pic_list.append(pic_dir + tmp_list[i])
            self.latent_list.append(latent_dir + tmp_list[i][:-3] + 'npy')

        self.pic_list = self.pic_list[:29984]
        '''for i in range(29984):
            print(self.pic_list[i])'''
        self.latent_list = self.latent_list[:29984]

        self.people_num = len(self.pic_list)

        self.type = 1
        self.bs = batch_size
        self.count = 0

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, index):
        p1 = random.randint(0, self.people_num - 1)
        p2 = p1

        if self.type == 0:
            # load pictures from the same folder
            pass
        else:
            # load pictures from different folders
            p2 = p1
            while p2 == p1:
                p2 = random.randint(0, self.people_num - 1)

        pic_id_dir = self.pic_list[p1]
        pic_att_dir = self.pic_list[p2]
        latent_id_dir = self.latent_list[p1]
        latent_att_dir = self.latent_list[p2]

        img_id = Image.open(pic_id_dir).convert('RGB')
        img_id = self.transformer(img_id)
        latent_id = np.load(latent_id_dir)
        latent_id = latent_id / np.linalg.norm(latent_id)
        latent_id = torch.from_numpy(latent_id)

        img_att = Image.open(pic_att_dir).convert('RGB')
        img_att = self.transformer(img_att)
        latent_att = np.load(latent_att_dir)
        latent_att = latent_att / np.linalg.norm(latent_att)
        latent_att = torch.from_numpy(latent_att)
        
        self.count += 1
        data_type = self.type
        if self.count == self.bs:
            self.type = 1 - self.type
            self.count = 0
        
        return img_id, img_att, latent_id, latent_att, data_type
        
    def __len__(self):
        return len(self.pic_list)