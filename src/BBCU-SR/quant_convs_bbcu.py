import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights,self.bias, stride=self.stride, padding=self.padding)

        return y

class BinaryConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(BinaryConv2d, self).__init__()

        self.move0 = LearnableBias(in_channels)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size//2),
        bias=bias)
        self.relu=RPReLU(out_channels)


    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out =self.relu(out)
        out = out + x
        return out

class BinaryUpConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False,upscale=2):
        super(BinaryUpConv2d, self).__init__()

        self.move0 = LearnableBias(in_channels)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size//2),
        bias=bias)
        self.relu=RPReLU(out_channels)
        self.upscale = upscale

        if self.upscale in [2, 3]:
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out =self.relu(out)
        out = self.pixel_shuffle(out)
        if self.upscale in [2, 3]:
            x = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        elif self.upscale == 4:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        out = out + x
        return out

class BinaryConv2dV2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(BinaryConv2dV2, self).__init__()

        self.move0 = LearnableBias(in_channels)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size//2),
        bias=bias)
        self.relu=RPReLU(out_channels)
        self.scale=out_channels//in_channels


    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.relu(out)
        x1 = torch.repeat_interleave(x,self.scale,1)
        
        x1 = torch.cat([x1,x[:,:1,:,:]],dim=1)
        out = x1+out
        return out

class BinaryBlock(nn.Module):
    def __init__(
            self,
            conv,
            n_feats,
            kernel_size,
            bias=True,
            bn=False,
            res_scale=1,
    ):
        super(BinaryBlock, self).__init__()
        m=[]
        for _ in range(2):
            m.append(conv(n_feats,n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res += x

        return res




