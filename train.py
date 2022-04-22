#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train.py
# Created Date: Monday December 27th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 22nd April 2022 10:49:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.backends import cudnn
import torch.utils.tensorboard as tensorboard

from util import util
from util.plot import plot_batch

from models.projected_model import fsModel
from data.data_loader_Swapping import GetLoader

def str2bool(v):
    return v.lower() in ('true')

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='simswap', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')       

        # for displays
        self.parser.add_argument('--use_tensorboard', type=str2bool, default='False')

        # for training
        self.parser.add_argument('--dataset', type=str, default="/path/to/VGGFace2", help='path to the face swapping dataset')
        self.parser.add_argument('--continue_train', type=str2bool, default='False', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224_test', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='10000', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')
        self.parser.add_argument('--Gdeep', type=str2bool, default='False')

        # for discriminators         
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_id', type=float, default=30.0, help='weight for id loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss') 

        self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--total_step", type=int, default=1000000, help='total training step')
        self.parser.add_argument("--log_frep", type=int, default=200, help='frequence for printing log information')
        self.parser.add_argument("--sample_freq", type=int, default=1000, help='frequence for sampling')
        self.parser.add_argument("--model_freq", type=int, default=10000, help='frequence for saving the model')

        


        self.isTrain = True
        
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt


if __name__ == '__main__':

    opt         = TrainOptions().parse()
    iter_path   = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    
    log_path = os.path.join(opt.checkpoints_dir, opt.name, 'summary')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    print("GPU used : ", str(opt.gpu_ids))

    
    cudnn.benchmark = True

    

    model = fsModel()

    model.initialize(opt)

    #####################################################
    if opt.use_tensorboard:
        tensorboard_writer  = tensorboard.SummaryWriter(log_path)
        logger              = tensorboard_writer
        
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D

    loss_avg        = 0
    refresh_count   = 0
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

    train_loader    = GetLoader(opt.dataset,opt.batchSize,8,1234)

    randindex = [i for i in range(opt.batchSize)]
    random.shuffle(randindex)

    if not opt.continue_train:
        start   = 0
    else:
        start   = int(opt.which_epoch)
    total_step  = opt.total_step
    import datetime
    print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    from util.logo_class import logo_class
    logo_class.print_start_training()
    model.netD.feature_network.requires_grad_(False)

    # Training Cycle
    for step in range(start, total_step):
        model.netG.train()
        for interval in range(2):
            random.shuffle(randindex)
            src_image1, src_image2  = train_loader.next()
            
            if step%2 == 0:
                img_id = src_image2
            else:
                img_id = src_image2[randindex]

            img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')
            latent_id       = model.netArc(img_id_112)
            latent_id       = F.normalize(latent_id, p=2, dim=1)
            if interval:
                
                img_fake        = model.netG(src_image1, latent_id)
                gen_logits,_    = model.netD(img_fake.detach(), None)
                loss_Dgen       = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                real_logits,_   = model.netD(src_image2,None)
                loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                loss_D          = loss_Dgen + loss_Dreal
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
            else:
                
                # model.netD.requires_grad_(True)
                img_fake        = model.netG(src_image1, latent_id)
                # G loss
                gen_logits,feat = model.netD(img_fake, None)
                
                loss_Gmain      = (-gen_logits).mean()
                img_fake_down   = F.interpolate(img_fake, size=(112,112), mode='bicubic')
                latent_fake     = model.netArc(img_fake_down)
                latent_fake     = F.normalize(latent_fake, p=2, dim=1)
                loss_G_ID       = (1 - model.cosin_metric(latent_fake, latent_id)).mean()
                real_feat       = model.netD.get_feature(src_image1)
                feat_match_loss = model.criterionFeat(feat["3"],real_feat["3"]) 
                loss_G          = loss_Gmain + loss_G_ID * opt.lambda_id + feat_match_loss * opt.lambda_feat
                

                if step%2 == 0:
                    #G_Rec
                    loss_G_Rec  = model.criterionRec(img_fake, src_image1) * opt.lambda_rec
                    loss_G      += loss_G_Rec

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                

        ############## Display results and errors ##########
        ### print out errors
        # Print out log info
        if (step + 1) % opt.log_frep == 0:
            # errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {
                "G_Loss":loss_Gmain.item(),
                "G_ID":loss_G_ID.item(),
                "G_Rec":loss_G_Rec.item(),
                "G_feat_match":feat_match_loss.item(),
                "D_fake":loss_Dgen.item(),
                "D_real":loss_Dreal.item(),
                "D_loss":loss_D.item()
            }
            if opt.use_tensorboard:
                for tag, value in errors.items():
                    logger.add_scalar(tag, value, step)
            message = '( step: %d, ) ' % (step)
            for k, v in errors.items():
                message += '%s: %.3f ' % (k, v)

            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

        ### display output images
        if (step + 1) % opt.sample_freq == 0:
            model.netG.eval()
            with torch.no_grad():
                imgs        = list()
                zero_img    = (torch.zeros_like(src_image1[0,...]))
                imgs.append(zero_img.cpu().numpy())
                save_img    = ((src_image1.cpu())* imagenet_std + imagenet_mean).numpy()
                for r in range(opt.batchSize):
                    imgs.append(save_img[r,...])
                arcface_112     = F.interpolate(src_image2,size=(112,112), mode='bicubic')
                id_vector_src1  = model.netArc(arcface_112)
                id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)

                for i in range(opt.batchSize):
                    
                    imgs.append(save_img[i,...])
                    image_infer = src_image1[i, ...].repeat(opt.batchSize, 1, 1, 1)
                    img_fake    = model.netG(image_infer, id_vector_src1).cpu()
                    
                    img_fake    = img_fake * imagenet_std
                    img_fake    = img_fake + imagenet_mean
                    img_fake    = img_fake.numpy()
                    for j in range(opt.batchSize):
                        imgs.append(img_fake[j,...])
                print("Save test data")
                imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
                plot_batch(imgs, os.path.join(sample_path, 'step_'+str(step+1)+'.jpg'))

        ### save latest model
        if (step+1) % opt.model_freq==0:
            print('saving the latest model (steps %d)' % (step+1))
            model.save(step+1)            
            np.savetxt(iter_path, (step+1, total_step), delimiter=',', fmt='%d')
    wandb.finish()