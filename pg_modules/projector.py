import torch
import torch.nn as nn
import timm
from pg_modules.blocks import FeatureFusionBlock


def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    # shapes
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4

    return scratch


def _make_efficientnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def calc_channels(pretrained, inp_res=224):
    channels = []
    tmp = torch.zeros(1, 3, inp_res, inp_res)

    # forward pass
    tmp = pretrained.layer0(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer1(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer2(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer3(tmp)
    channels.append(tmp.shape[1])

    return channels


def _make_projector(im_res, cout, proj_type, expand=False):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
    pretrained = _make_efficientnet(model)

    # determine resolution of feature maps, this is later used to calculate the number
    # of down blocks in the discriminators. Interestingly, the best results are achieved
    # by fixing this to 256, ie., we use the same number of down blocks per discriminator
    # independent of the dataset resolution
    im_res = 256
    pretrained.RESOLUTIONS = [im_res//4, im_res//8, im_res//16, im_res//32]
    pretrained.CHANNELS = calc_channels(pretrained)

    if proj_type == 0: return pretrained, None

    ### Build CCM
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)
    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1: return pretrained, scratch

    ### build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


class F_RandomProj(nn.Module):
    def __init__(
        self,
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.cout = cout
        self.expand = expand

        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, cout=self.cout, proj_type=self.proj_type, expand=self.expand)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x, get_features=False):
        # predict feature maps
        out0 = self.pretrained.layer0(x)
        out1 = self.pretrained.layer1(out0)
        out2 = self.pretrained.layer2(out1)
        out3 = self.pretrained.layer3(out2)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        backbone_features = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }
        if get_features:
            return backbone_features

        if self.proj_type == 0: return backbone_features

        out0_channel_mixed = self.scratch.layer0_ccm(backbone_features['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(backbone_features['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(backbone_features['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(backbone_features['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out, backbone_features
