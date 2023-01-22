from torch import nn as nn
from torch.nn import functional as F

from bbcu.utils.registry import ARCH_REGISTRY
from .arch_util import  default_init_weights, make_layer
from quant_convs_bbcu import BinaryBlock,BinaryConv2d,BinaryUpConv2d




@ARCH_REGISTRY.register()
class BBCUM(nn.Module):
    """

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4,img_range=1.):
        super(BBCUM, self).__init__()
        self.upscale = upscale
        self.img_range = img_range
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(BinaryBlock, num_block, conv=BinaryConv2d,
                    n_feats=num_feat,
                    kernel_size=3,
                    bias=False,
                    bn=False)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = BinaryUpConv2d(num_feat, num_feat * self.upscale * self.upscale, 3, False,upscale=upscale)
        elif self.upscale == 4:
            self.upconv1 = BinaryUpConv2d(num_feat, num_feat * 4, 3, False, upscale=2)
            self.upconv2 = BinaryUpConv2d(num_feat, num_feat * 4, 3, False, upscale=2)

        self.conv_hr = BinaryConv2d(num_feat, num_feat, 3)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        x = x * self.img_range
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.upconv1(out)
            out = self.upconv2(out)
        elif self.upscale in [2, 3]:
            out = self.upconv1(out)

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        out = out / self.img_range
        return out
