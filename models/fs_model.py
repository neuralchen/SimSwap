import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks

class SpecificNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(SpecificNorm, self).__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.mean = torch.from_numpy(self.mean).float().cuda()
        self.mean = self.mean.view([1, 3, 1, 1])

        self.std = np.array([0.229, 0.224, 0.225])
        self.std = torch.from_numpy(self.std).float().cuda()
        self.std = self.std.view([1, 3, 1, 1])

    def forward(self, x):
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])

        x = (x - mean) / std

        return x

class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, g_id, g_rec, g_mask, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, g_id, g_rec, g_mask, d_real, d_fake), flags) if f]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain

        device = torch.device("cuda:0")

        if opt.crop_size == 224:
            from .fs_networks import Generator_Adain_Upsample, Discriminator
        elif opt.crop_size == 512:
            from .fs_networks_512 import Generator_Adain_Upsample, Discriminator

        # Generator network
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        self.netG.to(device)

        # Id network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint
        self.netArc = self.netArc.to(device)
        self.netArc.eval()

        if not self.isTrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            return

        # Discriminator network
        if opt.gan_mode == 'original':
            use_sigmoid = True
        else:
            use_sigmoid = False
        self.netD1 = Discriminator(input_nc=3, use_sigmoid=use_sigmoid)
        self.netD2 = Discriminator(input_nc=3, use_sigmoid=use_sigmoid)
        self.netD1.to(device)
        self.netD2.to(device)

        #
        self.spNorm =SpecificNorm()
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        # load networks
        if opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # print (pretrained_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD1, 'D1', opt.which_epoch, pretrained_path)
            self.load_network(self.netD2, 'D2', opt.which_epoch, pretrained_path)



        if self.isTrain:
            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            self.criterionRec = nn.L1Loss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_ID', 'G_Rec', 'D_GP',
                                               'D_real', 'D_fake')

           # initialize optimizers

            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD1.parameters()) + list(self.netD2.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def _gradinet_penalty_D(self, netD, img_att, img_fake):
        # interpolate sample
        bs = img_fake.shape[0]
        alpha = torch.rand(bs, 1, 1, 1).expand_as(img_fake).cuda()
        interpolated = Variable(alpha * img_att + (1 - alpha) * img_fake, requires_grad=True)
        pred_interpolated = netD.forward(interpolated)
        pred_interpolated = pred_interpolated[-1]

        # compute gradients
        grad = torch.autograd.grad(outputs=pred_interpolated,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(pred_interpolated.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        loss_d_gp = torch.mean((grad_l2norm - 1) ** 2)

        return loss_d_gp

    def cosin_metric(self, x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def forward(self, img_id, img_att, latent_id, latent_att, for_G=False):
        loss_D_fake, loss_D_real, loss_D_GP = 0, 0, 0
        loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_ID, loss_G_Rec = 0,0,0,0,0

        img_fake = self.netG.forward(img_att, latent_id)
        if not self.isTrain:
            return img_fake
        img_fake_downsample = self.downsample(img_fake)
        img_att_downsample = self.downsample(img_att)



        # D_Fake
        fea1_fake = self.netD1.forward(img_fake.detach())
        fea2_fake = self.netD2.forward(img_fake_downsample.detach())
        pred_fake = [fea1_fake, fea2_fake]
        loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)


        # D_Feal
        fea1_real = self.netD1.forward(img_att)
        fea2_real = self.netD2.forward(img_att_downsample)
        pred_real = [fea1_real, fea2_real]
        fea_real = [fea1_real, fea2_real]
        loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        #print('=====================D_Real========================')

        # D_GP

        loss_D_GP = 0

        # G_GAN
        fea1_fake = self.netD1.forward(img_fake)
        fea2_fake = self.netD2.forward(img_fake_downsample)
        #pred_fake = [fea1_fake[-1], fea2_fake[-1]]
        pred_fake = [fea1_fake, fea2_fake]
        fea_fake = [fea1_fake, fea2_fake]
        loss_G_GAN = self.criterionGAN(pred_fake, True, for_discriminator=False)

        # GAN feature matching loss
        n_layers_D = 4
        num_D = 2
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(num_D):
                for j in range(0, len(fea_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(fea_fake[i][j],
                                                          fea_real[i][j].detach()) * self.opt.lambda_feat


        #G_ID
        img_fake_down = F.interpolate(img_fake, size=(112,112))
        img_fake_down = self.spNorm(img_fake_down)
        latent_fake = self.netArc(img_fake_down)
        loss_G_ID = (1 - self.cosin_metric(latent_fake, latent_id))
        #print('=====================G_ID========================')
        #print(loss_G_ID)

        #G_Rec
        loss_G_Rec = self.criterionRec(img_fake, img_att) * self.opt.lambda_rec

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_ID, loss_G_Rec, loss_D_GP, loss_D_real, loss_D_fake),
                img_fake]


    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD1, 'D1', which_epoch, self.gpu_ids)
        self.save_network(self.netD2, 'D2', which_epoch, self.gpu_ids)
        '''if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)'''

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


