import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from data.dataset_class import FaceDataSet
from torch.utils.data import DataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import cv2
from torchvision import transforms

def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0


detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10


dataset = FaceDataSet('people_list.txt', opt.batchSize)
data_loader = DataLoader(dataset, batch_size = opt.batchSize, shuffle=True)
dataset_size = len(data_loader)

device = torch.device("cuda:0")


model = create_model(opt)
visualizer = Visualizer(opt)

optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * 8608 + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

loss_avg = 0
refresh_count = 0

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, (img_id, img_att, latent_id, latent_att, data_type) in enumerate(data_loader):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # convert numpy to tensor
        img_id = img_id.to(device)
        img_att = img_att.to(device)
        latent_id = latent_id.to(device)
        latent_att = latent_att.to(device)


        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################

        losses, img_fake = model(img_id, img_att, latent_id, latent_att, for_G=True)

        # update Generator weights
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict['G_ID'] * opt.lambda_id
        if data_type[0] == 0:
            loss_G += loss_dict['G_Rec']

        optimizer_G.zero_grad()
        loss_G.backward(retain_graph=True)
        optimizer_G.step()

        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + loss_dict['D_GP']
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            '''visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])'''
            for i in range(img_id.shape[0]):
                if i == 0:
                    row1 = img_id[i]
                    row2 = img_att[i]
                    row3 = img_fake[i]
                else:
                    row1 = torch.cat([row1, img_id[i]], dim=2)
                    row2 = torch.cat([row2, img_att[i]], dim=2)
                    row3 = torch.cat([row3, img_fake[i]], dim=2)
            full = torch.cat([row1, row2, row3], dim=1).detach()
            full = full.permute(1, 2, 0)
            output = full.to('cpu')
            output = np.array(output)*255
            output = output[..., ::-1]
            cv2.imwrite('samples/step_'+str(total_steps)+'.jpg', output)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))