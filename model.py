from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from DISTS_pytorch import DISTS
import time
from einops import rearrange
import torchvision.transforms.functional as TF

# from HVI_transformer import RGB_HVI
import random, numbers
from dct_util import dct_2d, idct_2d, low_pass, low_pass_and_shuffle, high_pass
from measure import metrics
import matplotlib.pyplot as plt
import cv2
from loss import LossFunction
import os
from bsnDBSNl import DBSNl
from LLCaps import LLCaps, CWA
import util2
from timm.models.layers import trunc_normal_
from blocks import CBlock_ln, SwinTransformerBlock
from global_net import Global_pred
from PIL import Image
import torch
from logger import Logger
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
# from pytorch_msssim import ssim, ms_ssim
from complexPyTorch.complexLayers import ComplexConv2d, complex_relu
from tensorboardX import SummaryWriter
import math
from scipy.stats import wasserstein_distance
import torchvision.transforms as transforms


#
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import lpips
# from DISTS_pytorch import DISTS
# import time
# import random
# import matplotlib.pyplot as plt
# import cv2
# from IQA_pytorch import SSIM
# from .loss import LossFunction
# import os
# from einops import rearrange
#
# from .bsnDBSNl import DBSNl
# # from LLCaps import LLCaps,CWA
# from . import util2
# from timm.models.layers import trunc_normal_
# from .blocks import CBlock_ln, SwinTransformerBlock
# from .global_net import Global_pred
# from PIL import Image
# import torch
# from .logger import Logger
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# import numpy as np
# # from pytorch_msssim import ssim, ms_ssim
# from complexPyTorch.complexLayers import ComplexConv2d, complex_relu
# from tensorboardX import SummaryWriter
# import math
# from scipy.stats import wasserstein_distance
# import torchvision.transforms as transforms
def np2tensor(n: np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2, 0, 1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2, 0, 1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s' % (n.shape,))


def lpips2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")

    lpips_model = lpips.LPIPS(net="alex").to(device)  # alex
    # numpt to tensor
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # img1 = img1.transpose(1, 2, 0)
    # print(img1.shape) (600, 3, 400)
    img1 = np2tensor(img1).to(device)
    img2 = np2tensor(img2).to(device)
    # print(img1.shape) torch.Size([400, 600, 3])

    # img1 = Image.fromarray(np.uint8(img1))
    # img2 = Image.fromarray(np.uint8(img2))
    # img1 = preprocess(img1).unsqueeze(0).to(device)
    # img2 = preprocess(img2).unsqueeze(0).to(device)

    distance = lpips_model(img1, img2)
    return distance.item()


def dists2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")

    D = DISTS().to(device)
    # numpt to tensor
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img1 = img1.transpose(1, 2, 0)

    img1 = Image.fromarray(np.uint8(img1))
    img2 = Image.fromarray(np.uint8(img2))

    img1 = preprocess(img1).unsqueeze(0).to(device)
    img2 = preprocess(img2).unsqueeze(0).to(device)

    dists_value = D(img1, img2)

    return dists_value.item()


def ssim2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # if len(img1.shape) == 4:
    #     img1 = img1[0]
    # if len(img2.shape) == 4:
    #     img2 = img2[0]
    #
    # # tensor to numpy
    # if isinstance(img1, torch.Tensor):
    #     img1 = tensor2np(img1)
    # if isinstance(img2, torch.Tensor):
    #     img2 = tensor2np(img2)

    # numpy value cliping
    img2 = np.clip(img2, 0, 255)
    img1 = np.clip(img1, 0, 255)

    return structural_similarity(img1, img2, multichannel=True, data_range=255)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# ==pie-enhance======

class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        # r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image


# ==pie-enhance======

def frequency_loss(im1, im2):
    im1_fft = torch.fft.fftn(im1)
    im1_fft_real = im1_fft.real
    im1_fft_imag = im1_fft.imag
    im2_fft = torch.fft.fftn(im2)
    im2_fft_real = im2_fft.real
    im2_fft_imag = im2_fft.imag
    loss = 0
    for i in range(im1.shape[0]):
        real_loss = wasserstein_distance(im1_fft_real[i].reshape(
            im1_fft_real[i].shape[0] * im1_fft_real[i].shape[1] * im1_fft_real[i].shape[2]).cpu().detach(),
                                         im2_fft_real[i].reshape(im2_fft_real[i].shape[0] * im2_fft_real[i].shape[1] *
                                                                 im2_fft_real[i].shape[2]).cpu().detach())
        imag_loss = wasserstein_distance(im1_fft_imag[i].reshape(
            im1_fft_imag[i].shape[0] * im1_fft_imag[i].shape[1] * im1_fft_imag[i].shape[2]).cpu().detach(),
                                         im2_fft_imag[i].reshape(im2_fft_imag[i].shape[0] * im2_fft_imag[i].shape[1] *
                                                                 im2_fft_imag[i].shape[2]).cpu().detach())
        total_loss = real_loss + imag_loss
        loss += total_loss
    return torch.tensor(loss / (im1.shape[2] * im2.shape[3]))


#
# class Vgg16(nn.Module):
#     def __init__(self):
#         super(Vgg16, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, X):
#         h0 = F.relu(self.conv1_1(X), inplace=True)
#         h1 = F.relu(self.conv1_2(h0), inplace=True)
#         h2 = F.max_pool2d(h1, kernel_size=2, stride=2)
#
#         h3 = F.relu(self.conv2_1(h2), inplace=True)
#         h4 = F.relu(self.conv2_2(h3), inplace=True)
#         h5 = F.max_pool2d(h4, kernel_size=2, stride=2)
#
#         h6 = F.relu(self.conv3_1(h5), inplace=True)
#         h7 = F.relu(self.conv3_2(h6), inplace=True)
#         h8 = F.relu(self.conv3_3(h7), inplace=True)
#         h9 = F.max_pool2d(h8, kernel_size=2, stride=2)
#         h10 = F.relu(self.conv4_1(h9), inplace=True)
#         h11 = F.relu(self.conv4_2(h10), inplace=True)
#         conv4_3 = self.conv4_3(h11)
#         result = F.relu(conv4_3, inplace=True)
#
#         return result
# #
#
#
# def load_vgg16(model_dir):
#     """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     vgg = Vgg16()
#     vgg.cuda()
#     vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
#
#     return vgg
#
#
# def compute_vgg_loss(enhanced_result, input_high):
#     instance_norm = nn.InstanceNorm2d(512, affine=False)
#     vgg = load_vgg16("./model")
#     vgg.eval()
#     for param in vgg.parameters():
#         param.requires_grad = False
#     img_fea = vgg(enhanced_result)
#     target_fea = vgg(input_high)
#
#     loss = torch.mean((instance_norm(img_fea) - instance_norm(target_fea)) ** 2)
#
#     return loss
# class CalibrateNetwork(nn.Module):
#     def __init__(self, layers, channels):
#         super(CalibrateNetwork, self).__init__()
#         kernel_size = 3
#         dilation = 1
#         padding = int((kernel_size - 1) / 2) * dilation
#         self.layers = layers
#
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#         self.convs = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#         self.blocks = nn.ModuleList()
#         for i in range(layers):
#             self.blocks.append(self.convs)
#
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         fea = self.in_conv(input)
#         for conv in self.blocks:
#             fea = fea + conv(fea)
#
#         fea = self.out_conv(fea)
#         delta = input - fea
#
#         return delta
# #矫正 改进↓
class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers
        self.atten = IGAB(
            dim=channels, num_blocks=2, dim_head=channels, heads=1)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )  # 最初处理 残差

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )  # 循环 加残差
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # self.cou= nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=1)

    def forward(self, input, illu_fea):
        fea = self.in_conv(input)
        attfea = self.atten(fea, illu_fea)

        for conv in self.blocks:
            fea = conv(fea)
            fea = fea + conv(fea)
        catfea = torch.cat([fea, attfea], dim=1)
        fea = self.out_conv(catfea)
        delta = input - fea
        # d2=input+attfea#这 是干啥 矫正的？
        # delta=fea
        return delta


# class ResidualModule0(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(ResidualModule0, self).__init__()
#         self.Relu = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         residual = x
#         out0 = self.Relu(self.conv0(x))
#         out1 = self.Relu(self.conv1(out0))
#         out2 = self.Relu(self.conv2(out1))
#         out3 = self.Relu(self.conv3(out2))
#         out4 = self.Relu(self.conv4(out3))
#         out = self.Relu(self.conv(residual))
#
#         final_out = torch.cat((out, out4), dim=1)
#
#         return final_out
#
#
# class ResidualModule1(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(ResidualModule1, self).__init__()
#         self.Relu = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         residual = x
#         out0 = self.Relu(self.conv0(x))
#         out1 = self.Relu(self.conv1(out0))
#         out2 = self.Relu(self.conv2(out1))
#         out3 = self.Relu(self.conv3(out2))
#         out4 = self.Relu(self.conv4(out3))
#         out = self.Relu(self.conv(residual))
#
#         final_out = torch.cat((out, out4), dim=1)
#
#         return final_out
#
#
# class ResidualModule2(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(ResidualModule2, self).__init__()
#         self.Relu = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         residual = x
#         out0 = self.Relu(self.conv0(x))
#         out1 = self.Relu(self.conv1(out0))
#         out2 = self.Relu(self.conv2(out1))
#         out3 = self.Relu(self.conv3(out2))
#         out4 = self.Relu(self.conv4(out3))
#         out = self.Relu(self.conv(residual))
#
#         final_out = torch.cat((out, out4), dim=1)
#
#         return final_out
#
#
# class ResidualModule3(nn.Module):
#     def __init__(self, channel=64, kernel_size=3):
#         super(ResidualModule3, self).__init__()
#         self.Relu = nn.LeakyReLU()
#
#         self.conv0 = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(channel, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channel * 2, channel * 4, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channel * 4, channel * 2, kernel_size, stride=1, padding=1, padding_mode='replicate')
#         self.conv4 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
#
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         residual = x
#         out0 = self.Relu(self.conv0(x))
#         out1 = self.Relu(self.conv1(out0))
#         out2 = self.Relu(self.conv2(out1))
#         out3 = self.Relu(self.conv3(out2))
#         out4 = self.Relu(self.conv4(out3))
#         out = self.Relu(self.conv(residual))
#
#         final_out = torch.cat((out, out4), dim=1)
#
#         return final_out
#
#
# class Resblock(nn.Module):
#     def __init__(self, channels, kernel_size=3, stride=1):
#         super(Resblock, self).__init__()
#
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = (stride, stride)
#         self.activation = nn.LeakyReLU(True)
#
#         sequence = list()
#
#         sequence += [
#             nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='replicate'),
#             nn.LeakyReLU(),
#             nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1, padding_mode='replicate'),
#         ]
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, x):
#
#         residual = x
#         output = self.activation(self.model(x) + residual)
#
#         return output
#
#
# class complex_net(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(complex_net, self).__init__()
#
#         self.complex_conv0 = ComplexConv2d(in_channels, out_channels*2, kernel_size=3, stride=1, padding=1)
#         self.complex_conv1 = ComplexConv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1)
#         self.complex_conv2 = ComplexConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#
#         residual = x
#         out0 = complex_relu(self.complex_conv0(x))
#         out1 = complex_relu(self.complex_conv1(out0))
#         out2 = complex_relu(self.complex_conv2(out1))
#         output = residual + out2
#
#         return output
#
#
# class feature_block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(feature_block, self).__init__()
#
#         self.resblock0 = Resblock(in_channels)
#         self.complex_block = complex_net(out_channels, out_channels)
#         self.resblock1 = Resblock(out_channels)
#
#     def forward(self, x):
#
#         residual = x
#         out0 = self.resblock0(x)
#         fft_out0 = torch.fft.rfftn(out0)
#         out1 = self.complex_block(fft_out0)
#         ifft_out1 = torch.fft.irfftn(out1)
#         out2 = self.resblock1(ifft_out1)
#
#         output = residual + out2
#
#         return output
#
#
# class fft_processing(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(fft_processing, self).__init__()
#         self.complex_conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#
#         self.complex_block0 = complex_net(out_channels, out_channels)
#         self.complex_block1 = complex_net(out_channels, out_channels)
#
#         self.complex_conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#
#         conv1_out = complex_relu(self.complex_conv1(x))
#         complex_block_out0 = self.complex_block0(conv1_out)
#         complex_block_out1 = self.complex_block1(complex_block_out0)
#         conv2_out = complex_relu(self.complex_conv2(complex_block_out1))
#
#         return conv2_out


# class Local_pred(nn.Module):
#     def __init__(self, dim=16, number=4, type='ccc'):
#         super(Local_pred, self).__init__()
#         # initial convolution
#         self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
#         self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         # main blocks
#         block = CBlock_ln(dim)
#         block_t = SwinTransformerBlock(dim)  # head number
#         if type == 'ccc':
#             # blocks1, blocks2 = [block for _ in range(number)], [block for _ in range(number)]
#             blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
#             blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
#         elif type == 'ttt':
#             blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
#         elif type == 'cct':
#             blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
#         #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
#         self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
#         self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
#
#     def forward(self, img):
#         img1 = self.relu(self.conv1(img))
#         mul = self.mul_blocks(img1)
#         add = self.add_blocks(img1)
#
#         return mul, add


# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, stride=1, padding=1, groups=1)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type == 'ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == 'ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == 'cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        # print(img)#[[0.0863, 0.1176, 0.1216,  ..., 0.1137, 0.1176, 0.1098],
        # img1 = self.relu(self.conv1(img))
        # e-01   ====10 ^ -01
        img11 = self.conv1(img)
        # print(img11) #[ 9.9695e-03, -2.3828e-03, -6.4886e-03,  ..., -3.4269e-03,
        img1 = self.relu(img11)
        # print(img1) #[ 9.9695e-03, -4.7655e-04, -1.2977e-03,  ..., -6.8539e-04,
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        # print(mul) # 5.4162e-02,  3.9797e-02,  3.8274e-02,  ...,  3.9620e-02,
        add = self.add_blocks(img1) + img1

        # print(add) #0.0374,  0.0076,  0.0074,  ...,  0.0090,  0.0096,  0.0085],
        mul = self.mul_end(mul)
        # print(mul) #[0., 0., 0.,  ..., 0., 0., 0.],
        add = self.add_end(add)
        # print(add) #[0.9997, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 0.9996]
        return mul, add


#
# class EnhanceNetwork(nn.Module):
#     def __init__(self, layers, channels):
#         super(EnhanceNetwork, self).__init__()
#
#         kernel_size = 3
#         dilation = 1
#         padding = int((kernel_size - 1) / 2) * dilation
#
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.ReLU()
#         )
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#         self.blocks = nn.ModuleList()
#         for i in range(layers):
#             self.blocks.append(self.conv)
#
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         fea = self.in_conv(input)
#         for conv in self.blocks:
#             fea = fea + conv(fea)
#         fea = self.out_conv(fea)
#
#         illu = fea + input
#         illu = torch.clamp(illu, 0.0001, 1)
#
#         return illu


class Illumination_Estimator(nn.Module):  # 视网膜电流发生器采用了所提出的由照明估计器组成的ORF
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.depth_conv3 = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=3, padding=2, bias=True, groups=n_fea_in)
        self.conv3 = nn.Conv2d(n_fea_middle * 2, n_fea_middle, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)  # 平均通道=照明先验
        # stx()
        input = torch.cat([img, mean_c], dim=1)
        # input=img
        x_1 = self.conv1(input)
        illu_fea1 = self.depth_conv(x_1)
        illu_fea2 = self.depth_conv(x_1)
        illu_fea3 = torch.cat([illu_fea1, illu_fea2], dim=1)
        illu_fea = self.conv3(illu_fea3)  # 特征
        illu_map = self.conv2(illu_fea)  # 亮度映射直接乘原图能得到亮图

        return illu_fea, illu_map


#
# class Illumination_Estimator(nn.Module):  # 视网膜电流发生器采用了所提出的由照明估计器组成的ORF
#     def __init__(
#             self, n_fea_middle, n_fea_in=4, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
#         super(Illumination_Estimator, self).__init__()
#
#         self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
#
#         self.depth_conv = nn.Conv2d(
#             n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
#
#         self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
#
#     def forward(self, img):
#         # img:        b,c=3,h,w
#         # mean_c:     b,c=1,h,w
#
#         # illu_fea:   b,c,h,w
#         # illu_map:   b,c=3,h,w
#
#         mean_c = img.mean(dim=1).unsqueeze(1)  # 平均通道=照明先验
#         # stx()
#         input = torch.cat([img, mean_c], dim=1)
#         # input=img
#         x_1 = self.conv1(input)
#         illu_fea = self.depth_conv(x_1)  # 特征
#         illu_map = self.conv2(illu_fea)  # 亮度映射直接乘原图能得到亮图
#
#
#         return illu_fea, illu_map


class LightNet(nn.Module):  # 照度
    def __init__(self, nf):  # nf=32
        super(LightNet, self).__init__()
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.in2 = nn.InstanceNorm2d(nf, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=True)
        self.in3 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.in4 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=True)
        self.in5 = nn.InstanceNorm2d(nf * 4, affine=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=True)
        self.in6 = nn.InstanceNorm2d(nf * 4, affine=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(nf * 4, nf * 2, 1, 1, 0, bias=True)
        self.in7 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.in8 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(nf * 2, nf, 1, 1, 0, bias=True)
        self.in9 = nn.InstanceNorm2d(nf, affine=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.in10 = nn.InstanceNorm2d(nf, affine=True)
        self.relu10 = nn.ReLU(inplace=True)

    def forward(self, x):
        out2 = self.relu2(self.in2(self.conv2(x)))

        out3 = self.relu3(self.in3(self.conv3(out2)))
        out4 = self.relu4(self.in4(self.conv4(out3)))

        out5 = self.relu5(self.in5(self.conv5(out4)))
        out6 = self.relu6(self.in6(self.conv6(out5)))

        up1 = F.interpolate(out6, size=[out4.size()[2], out4.size()[3]], mode='bilinear')
        out7 = self.relu7(self.in7(self.conv7(up1)))
        out8 = self.relu8(self.in8(self.conv8(out7 + out4)))

        up2 = F.interpolate(out8, size=[out2.size()[2], out2.size()[3]], mode='bilinear')
        out9 = self.relu9(self.in9(self.conv9(up2)))
        out10 = self.relu10(self.in10(self.conv10(out9 + out2)))

        return out10


def transform_invert(img_):
    img_ = img_.squeeze(0).transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = img_.detach().cpu().numpy() * 255.0

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class IG_MSA(nn.Module):  # MSA使用ORF捕获的照明表示来指导自注意的计算。照度引导注意力
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):  # 两个接收的
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn  # cheng 16 36
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):  # IGT的基本单元是IGAB，它由两层归一化（LN）、一个IG-MSA和一个前馈网络（FFN）组成。(
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),  # 注意力
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)  # 将张量x的维度顺序进行调整,并将结果存储在一个新的张量 统一到一个维度  归一化
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x  # 处理注意力
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)  # 转回之前维度 二个维度 不懂
        return out


# y应用原始
# class EnhanceNetwork(nn.Module):
#     def __init__(self, layers, channels):
#         super(EnhanceNetwork, self).__init__()
#
#         kernel_size = 3
#         dilation = 1
#         padding = int((kernel_size - 1) / 2) * dilation
#
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.ReLU()
#         )
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )  # 提特征
#
#         self.blocks = nn.ModuleList()
#         for i in range(layers):
#             self.blocks.append(self.conv)
#
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         fea = self.in_conv(input)
#         for conv in self.blocks:
#             fea = fea + conv(fea)  # 循环两次 残差
#         fea = self.out_conv(fea)
#
#         illu = fea + input  # 加input
#         illu = torch.clamp(illu, 0.0001, 1)  # 获取i
#
#         return illu

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
        # 添加可学习的空间缩放因子
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        norm_x = to_4d(self.body(to_3d(x)), h, w)
        return norm_x * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        # 多尺度分支
        self.project_in1 = nn.Conv2d(dim, hidden_features, 1, bias=bias)
        self.project_in2 = nn.Conv2d(dim, hidden_features, 3, padding=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, padding=1, groups=hidden_features * 2, bias=bias)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_features * 2, hidden_features // 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_features // 2, hidden_features * 2, 1),
            nn.Sigmoid()
        )

        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x1 = self.project_in1(x)  # 1x1分支
        x2 = self.project_in2(x)  # 3x3分支
        x = torch.cat([x1, x2], dim=1)

        x = self.dwconv(x)
        g = self.gate(x)  # 自适应门控
        x1, x2 = (x * g).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 位置偏置
        self.pos_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

        self.save_dir = 'feature_maps'
        os.makedirs(self.save_dir, exist_ok=True)

    def save_feature_map(self,feature_map, save_dir='feature_heatmaps', prefix='layer'):
        """
        保存特征图为热力图（Heatmap）
        :param feature_map: Tensor, 形状为 (B, C, H, W)
        :param save_dir: 图像保存目录
        :param prefix: 文件名前缀
        """
        os.makedirs(save_dir, exist_ok=True)
        feature_map = feature_map.detach().cpu()

        # 只处理 batch 中第一个样本
        fmap = feature_map[0]

        for i in range(min(8, fmap.shape[0])):  # 取前8个通道做示例
            channel = fmap[i]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)  # 归一化

            plt.figure(figsize=(4, 4))
            plt.axis('off')
            plt.imshow(channel.numpy(), cmap='jet')  # 使用 jet 颜色映射显示为热力图
            plt.savefig(os.path.join(save_dir, f'{prefix}_heatmap_ch{i}.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
    # def _save_feature_map(self, tensor, name, step=0):
    #     """保存特征图到本地"""
    #     # 通道压缩（取前3个通道或均值）
    #     if tensor.shape[1] >= 3:  # RGB可视化
    #         vis_tensor = tensor[0, :3].detach().cpu()
    #     else:  # 单通道特征
    #         vis_tensor = tensor.mean(dim=1, keepdim=True)[0].detach().cpu()
    #
    #     # 归一化到[0,1]
    #     vis_tensor = (vis_tensor - vis_tensor.min()) / (vis_tensor.max() - vis_tensor.min() + 1e-6)
    #
    #     # 保存为图片
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(vis_tensor.permute(1, 2, 0).squeeze() if vis_tensor.shape[0] == 1
    #                else vis_tensor.permute(1, 2, 0))
    #     plt.title(f"{name} (shape: {tensor.shape})")
    #     plt.colorbar()
    #     plt.savefig(f"{self.save_dir}/step{step}_{name}.png")
    #     plt.close()
    def forward(self, x):

        # self.save_feature_map(x, save_dir='feature_maps/x', prefix='conv1')
        b, c, h, w = x.shape

        # 通道注意力
        ca = self.channel_attn(x)


        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature + self.pos_bias
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 应用通道注意力
        out = out * ca

        # self.save_feature_map(out, save_dir='feature_maps/out', prefix='conv1')
        return self.project_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        # 双路径注意力机制
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        # 改进的FFN
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        # 通道注意力模块
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1, bias=bias),
            nn.Sigmoid()
        )

        # 动态残差权重
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.ffn_scale = nn.Parameter(torch.ones(1))
        self.channel_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 主路径
        attn_out = self.attn(self.norm1(x))

        # 通道注意力
        channel_weight = self.channel_attn(x)

        # 特征融合
        x = x + self.attn_scale * attn_out
        x = x * (1 + self.channel_scale * channel_weight)

        # FFN路径
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.ffn_scale * ffn_out

        return x


class IGAB(nn.Module):  # IGT的基本单元是IGAB，它由两层归一化（LN）、一个IG-MSA和一个前馈网络（FFN）组成。(
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),  # 注意力
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)  # 将张量x的维度顺序进行调整,并将结果存储在一个新的张量 统一到一个维度  归一化
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x  # 处理注意力
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)  # 转回之前维度 二个维度 不懂
        return out


class CurveCALayer(nn.Module):
    def __init__(self, channel):
        super(CurveCALayer, self).__init__()

        self.n_curve = 3
        self.relu = nn.ReLU(inplace=False)

        # 多尺度特征提取
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, 3, 1, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AvgPool2d(1),
                nn.Conv2d(channel, channel, 3, 1, 1),
                nn.ReLU(inplace=True)
            )
        ])

        # 动态曲线参数预测
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 5, 1, 2),  # 融合多尺度特征
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, groups=channel),  # 深度可分离卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1),  # 输出3条曲线的参数
            nn.Tanh()  # 限制参数范围
        )

    def forward(self, x):
        # 多尺度特征融合
        features = [conv(x) for conv in self.multi_scale]
        # print(x.shape)
        # print(features[0].shape)
        # print([x].[0].shape)
        features = torch.cat(features + [x], dim=1)

        # 动态预测曲线参数
        a = self.predict_a(features)  # [B,3,H,W]
        x = self.relu(x) - self.relu(x - 1)
        # 改进的曲线调整公式
        # x = torch.clamp(x, 0, 1)  # 确保输入在[0,1]范围
        for i in range(3):
            # 自适应调整曲线：S形曲线+线性补偿
            x = x + a[:, i:i + 1] * (x - x.pow(2)) * (1 + 0.5 * x)

            # 柔性截断替代硬截断
        return 0.5 * (torch.tanh(5 * (x - 0.5)) + 1)  # 平滑过渡到[0,1]范围


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        # self.curve = CurveCALayer(channels)
        self.illumap = nn.Conv2d(3, 1, 1)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )  # 提特征

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        mean_c = input.mean(dim=1).unsqueeze(1)  # 平均通道=照明先验
        # stx()
        input1 = torch.cat([input, mean_c], dim=1)
        fea = self.in_conv(input1)
        for conv in self.blocks:
            fea = fea + conv(fea)
            # fea = self.curve(fea)  # 循环两次 残差
        fea = self.out_conv(fea)

        illu = fea + input  # 加input
        # illu = fea + torch.mean(input, dim=1, keepdim=True)
        illu = torch.clamp(illu, 0.0001, 1)  # 获取i
        # illu = self.illumap(illu)

        return illu, illu


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class N_net(nn.Module):
    def __init__(self, num=64, num_heads=1, num_blocks=2, inp_channels=3, out_channels=3, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super(N_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[
            TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input):
        x = self.patch_embed(input)
        x = self.encoder(x)
        out = self.output(x)

        return torch.sigmoid(out) + input

#
class L_net(nn.Module):
    def __init__(self, num=48, num_heads=1, num_blocks=2, inp_channels=3, out_channels=1, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super(L_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[
            TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, input):
        # mean_c = input.mean(dim=1).unsqueeze(1)  # 平均通道=照明先验
        # # stx()
        # input1 = torch.cat([input, mean_c], dim=1)
        out = self.patch_embed(input)
        out = self.encoder(out)
        out = self.output(out)
        out=torch.sigmoid(out) + torch.mean(input, dim=1, keepdim=True)
        return out
#         return torch.sigmoid(out)
# class L_net(nn.Module):
#     def __init__(self, num=48, num_heads=1, num_blocks=2, inp_channels=3, out_channels=1, ffn_expansion_factor=2.66,
#                  bias=False, LayerNorm_type='WithBias'):
#         super(L_net, self).__init__()
#         self.patch_embed = OverlapPatchEmbed(inp_channels, num)
#         self.encoder = nn.Sequential(*[
#             TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                              LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)])
#         self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#
#     def forward(self, input):
#         out = self.patch_embed(input)
#         out = self.encoder(out)
#         out = self.output(out)
#         L = torch.sigmoid(out)
#         return torch.clamp(L, 1e-3, 1.0)  # 防止除0


class L_enhance_net(nn.Module):
    def __init__(self, in_channel=1, num=32, num_heads=1, num_blocks=2, ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super(L_enhance_net, self).__init__()
        self.patch_embed = OverlapPatchEmbed(in_channel, num)
        self.encoder = nn.Sequential(*[
            TransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU()
        )
        self.lin = nn.AdaptiveAvgPool2d(1)
        self.tail = nn.Sequential(
            nn.Conv2d(num, num // 2, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(num // 2, 1, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

    def forward(self, input):
        out = self.head(input)
        # out = self.patch_embed(input)
        # out = self.encoder(out)
        # print(out)
        out = self.lin(out)
        # print(out)
        out = self.tail(out)
        # print(out)
        return out


class crossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(crossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        # print(x.shape, y)

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q(y)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # print(attn.shape)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class crossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(crossTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = crossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.skip_scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.skip_scale2 = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, input):
        x = input[0]
        y = input[1]
        x = x * self.skip_scale + self.attn(self.norm1(x), y)
        x = x * self.skip_scale2 + self.ffn(self.norm2(x))

        return [x, y]


class R_net(nn.Module):
    def __init__(self, num=64, num_heads=1, num_blocks=2, inp_channels=3, out_channels=3, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super(R_net, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, num)
        self.encoder = nn.Sequential(*[
            crossTransformerBlock(dim=num, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                  LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Conv2d(num, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input, fea):
        x = self.patch_embed(input)
        x, _ = self.encoder([x, fea])
        # x =self.encoder(x)
        out = self.output(x)
        return torch.sigmoid(out) + input
        # return torch.sigmoid(out)


class F_light_prior_estimater(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=16, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(F_light_prior_estimater, self).__init__()

        self.conv1 = nn.Conv2d(16, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        _, _, h, w = img.shape
        depth = min(h, w) // 10
        mean_c = img.mean(dim=1).unsqueeze(1)  # 先验
        # stx()
        ###dct
        f0 = dct_2d(img, norm='ortho')
        f1 = low_pass(f0, 3 * depth).cuda()
        f2 = low_pass_and_shuffle(f0, depth).cuda()
        f3 = high_pass(low_pass(f0, 4 * depth), 2 * depth).cuda()
        f4 = high_pass(f0, 5 * depth).cuda()
        ff = torch.cat([f1, f2, f3, f4], dim=1)  #
        ff = idct_2d(ff, norm='ortho')  # ？统计域

        input = torch.cat([img, mean_c, ff], dim=1)  # estimator

        x_1 = self.conv1(input)

        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class BasicBlock(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class LightBackbone(nn.Sequential):
    r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).

    Args:
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to use an extra pooling layer at the end
            of the backbone. Default: False.
        n_base_feats (int, optional): Channel multiplier. Default: 8.
    """

    def __init__(self,
                 input_resolution=256,
                 extra_pooling=False,
                 n_base_feats=8,
                 **kwargs) -> None:
        body = [BasicBlock(3, n_base_feats, stride=2, norm=True)]
        n_feats = n_base_feats
        for _ in range(3):
            body.append(
                BasicBlock(n_feats, n_feats * 2, stride=2, norm=True))
            n_feats = n_feats * 2
        body.append(BasicBlock(n_feats, n_feats, stride=2))
        body.append(nn.Dropout(p=0.5))
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = n_feats * (
            4 if extra_pooling else (input_resolution // 32) ** 2)

    def forward(self, imgs):

        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
                             mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


#
# class LUT1DGenerator(nn.Module):
#     r"""The 1DLUT generator module.
#
#     Args:
#         n_colors (int): Number of input color channels.
#         n_vertices (int): Number of sampling points.
#         n_feats (int): Dimension of the input image representation vector.
#         color_share (bool, optional): Whether to share a single 1D LUT across
#             three color channels. Default: False.
#     """
#
#     def __init__(self, n_colors, n_vertices, n_feats, color_share=False) -> None:
#         super().__init__()
#         repeat_factor = n_colors if not color_share else 1
#         self.lut1d_generator = nn.Linear(
#             n_feats, n_vertices * repeat_factor)
#
#         self.n_colors = n_colors
#         self.n_vertices = n_vertices
#         self.color_share = color_share
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         lut1d = self.lut1d_generator(x).view(
#             x.shape[0], -1, self.n_vertices)
#         if self.color_share:
#             lut1d = lut1d.repeat_interleave(self.n_colors, dim=1)
#         lut1d = lut1d.sigmoid()
#
#         return lut1d


# def lut_transform(imgs, luts):
#     # img (b, 3, h, w), lut (b, c, m, m, m)
#
#     # normalize pixel values
#     imgs = (imgs - .5) * 2.
#     # reshape img to grid of shape (b, 1, h, w, 3)
#     grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)
#
#     # after gridsampling, output is of shape (b, c, 1, h, w)
#     outs = F.grid_sample(luts, grids,
#         mode='bilinear', padding_mode='border', align_corners=True)
#     # remove the extra dimension
#     outs = outs.squeeze(2)
#     return outs
def apply_lut1d_to_image(img, lut1d):
    """
    img: (B, 3, H, W), range [0, 1]
    lut1d: (B, 3, N), each channel's LUT, range [0, 1]
    """
    B, C, H, W = img.shape
    N = lut1d.shape[-1]

    # Expand LUT to match image shape
    img_flat = img.view(B, C, -1)  # (B, 3, H*W)
    img_scaled = img_flat * (N - 1)  # scale to index
    idx_low = torch.floor(img_scaled).long().clamp(0, N - 2)  # lower index
    idx_high = idx_low + 1
    weight_high = img_scaled - idx_low.float()
    weight_low = 1. - weight_high

    # Gather LUT values
    lut_flat = lut1d  # (B, 3, N)
    val_low = torch.gather(lut_flat, 2, idx_low)
    val_high = torch.gather(lut_flat, 2, idx_high)

    # Interpolate
    out = val_low * weight_low + val_high * weight_high
    out = out.view(B, C, H, W)
    return out


#
# class LUT1DGenerator(nn.Module):
#     def __init__(self, n_colors, n_vertices, n_feats, color_share=False) -> None:
#         super().__init__()
#         repeat_factor = n_colors if not color_share else 1
#         self.lut1d_generator = nn.Linear(n_feats, n_vertices * repeat_factor)
#
#         self.n_colors = n_colors
#         self.n_vertices = n_vertices
#         self.color_share = color_share
#
#         # 恒等映射：从 0 到 1 均匀分布
#         self.register_buffer('identity', torch.linspace(0, 1, n_vertices).view(1, 1, -1))
#
#         # for p in self.lut1d_generator.parameters():
#         #     nn.init.normal_(p, mean=0.0, std=0.2)  # 默认 std=0.02，太小了
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         delta = self.lut1d_generator(x).view(x.shape[0], -1, self.n_vertices)
#         if self.color_share:
#             delta = delta.repeat_interleave(self.n_colors, dim=1)
#
#         # 恒等 LUT + 可学习扰动（通过 tanh 控制振幅）
#         lut1d = self.identity + 0.3 * delta.tanh()
#         lut1d = lut1d.clamp(0, 1)  # 限制范围
#         # print(lut1d.shape)
#         return lut1d
#

class LUT1DGenerator(nn.Module):
    r"""The 1DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points.
        n_feats (int): Dimension of the input image representation vector.
        color_share (bool, optional): Whether to share a single 1D LUT across
            three color channels. Default: False.
    """

    def __init__(self, n_colors, n_vertices, n_feats, color_share=False) -> None:
        super().__init__()
        repeat_factor = n_colors if not color_share else 1
        self.lut1d_generator = nn.Linear(
            n_feats, n_vertices * repeat_factor)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.color_share = color_share

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        lut1d = self.lut1d_generator(x).view(
            x.shape[0], -1, self.n_vertices)
        if self.color_share:
            lut1d = lut1d.repeat_interleave(self.n_colors, dim=1)
        lut1d = lut1d.sigmoid()

        return lut1d
#
#
# class MultiLUT1DGenerator(nn.Module):
#     def __init__(self, n_vertices, n_feats):
#         super().__init__()
#         self.lut_dark = nn.Linear(n_feats, n_vertices * 3)
#         self.lut_mid = nn.Linear(n_feats, n_vertices * 3)
#         self.lut_bright = nn.Linear(n_feats, n_vertices * 3)
#         self.n_vertices = n_vertices
#
#     def forward(self, feat):
#         B = feat.shape[0]
#         lut_dark = self.lut_dark(feat).view(B, 3, self.n_vertices).sigmoid()
#         lut_mid = self.lut_mid(feat).view(B, 3, self.n_vertices).sigmoid()
#         lut_bright = self.lut_bright(feat).view(B, 3, self.n_vertices).sigmoid()
#
#         return lut_dark, lut_mid, lut_bright
#
#
# def apply_multi_lut1d(img, luts, luminance):
#     """
#     img: [B, 3, H, W], luts: (lut_dark, lut_mid, lut_bright): each [B, 3, N]
#     luminance: [B, 1, H, W] from input image
#     """
#     B, C, H, W = img.shape
#     N = luts[0].shape[-1]
#
#     # Interpolation indices
#     img_flat = img.view(B, C, -1)
#     img_scaled = img_flat * (N - 1)
#     idx_low = torch.floor(img_scaled).long().clamp(0, N - 2)
#     idx_high = idx_low + 1
#     weight_high = img_scaled - idx_low.float()
#     weight_low = 1. - weight_high
#
#     # LUT gather
#     def interp_lut(lut):
#         val_low = torch.gather(lut, 2, idx_low)
#         val_high = torch.gather(lut, 2, idx_high)
#         return val_low * weight_low + val_high * weight_high
#
#     img_dark = interp_lut(luts[0])
#     img_mid = interp_lut(luts[1])
#     img_bright = interp_lut(luts[2])
#
#     # reshape to [B, C, H, W]
#     img_dark = img_dark.view(B, C, H, W)
#     img_mid = img_mid.view(B, C, H, W)
#     img_bright = img_bright.view(B, C, H, W)
#
#     # brightness mask weights
#     w_dark = (luminance < 0.3).float()
#     w_mid = ((luminance >= 0.3) & (luminance < 0.7)).float()
#     w_bright = (luminance >= 0.7).float()
#
#     # fuse
#     output = img_dark * w_dark + img_mid * w_mid + img_bright * w_bright
#     # visualize_multi_luts(luts, fused_lut, step=current_step)
#     return output
class LearnableLUT1D(nn.Module):
    def __init__(self,
                 n_vertices_1d=17,
                 lut1d_color_share=False,
                 n_colors=3,
                 ):
        super().__init__()

        self.n_colors = n_colors
        self.n_vertices_1d = n_vertices_1d

        self.fp16_enabled = False
        self.init_weights()
        self.backbone = LightBackbone()
        self.lut1d_generator = LUT1DGenerator(
            n_colors, n_vertices_1d, self.backbone.out_channels,
            color_share=lut1d_color_share)
        # self.multi_lut=MultiLUT1DGenerator( n_vertices_1d, self.backbone.out_channels)
    def init_weights(self):
        r"""Init weights for models.

        For the backbone network and the 3D LUT generator, we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
        """

        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(special_initilization)
        # self.register_buffer('identity', torch.linspace(0, 1,  self.n_vertices_1d ).view(1, 1, -1))  # shape: (1,1,n)

    def forward(self, imgs):
        # context vector: (b, f)

        codes = self.backbone(imgs)

        lut1d = self.lut1d_generator(codes)
        enhanced = apply_lut1d_to_image(imgs, lut1d)  # 生成1d

        # lut1d = self.multi_lut(codes)
        # # print(lut1d.shape)
        # # lum = imgs.mean(dim=1, keepdim=True)
        # enhanced = apply_multi_lut1d(imgs, lut1d, L)

        return enhanced, lut1d


# def plot_rgb_luts(lut_param, save_path="rgb_lut.png"):
#     """
#     支持 shape: [1, 3, N] 或 [B, 3, N]
#     """
#     lut = lut_param.detach().cpu().squeeze()
#     if len(lut.shape) == 1:
#         lut = lut.unsqueeze(0)
#     n_bins = lut.shape[-1]
#     x = torch.linspace(0, 1, n_bins)
#     colors = ['r', 'g', 'b']
#     plt.figure()
#     for i in range(lut.shape[0]):
#         plt.plot(x, lut[i], label=f'LUT-{colors[i]}', color=colors[i])
#     plt.plot(x, x, '--', label="Identity", alpha=0.5)
#     plt.legend(); plt.grid(True); plt.tight_layout()
#     plt.savefig(save_path); plt.close()
# def visualize_lut_1d(lut_param, save_path="lut_curve.png"):
#     lut = lut_param.detach().cpu().squeeze(0).numpy()  # shape: [3, N]
#     n_bins = lut.shape[-1]
#     x = np.linspace(0, 1, n_bins)
#     colors = ['r', 'g', 'b']
#     plt.figure()
#     for i in range(3):
#         plt.plot(x, lut[i], label=f'LUT-{colors[i]}', color=colors[i])
#     plt.plot(x, x, '--', label="Identity", alpha=0.5)
#     plt.legend(); plt.grid(True); plt.tight_layout()
#     plt.savefig(save_path); plt.close()
#
# def visualize_lut_1d(lut_param: torch.Tensor, save_path="lut_curve.png"):
#     """
#     可视化 1D LUT 学到的曲线，并与 identity 对比。
#
#     Args:
#         lut_param: [1, n_bins] 的 torch.Parameter（LUT 参数）
#         save_path: 保存图像的路径
#         step: 当前 step（用于显示）
#     """
#     lut = lut_param.detach().cpu().numpy()[0]
#     n_bins = len(lut)
#     x = np.linspace(0, 1, n_bins)
#
#     plt.figure(figsize=(6, 5))
#     plt.plot(x, lut, label="Learned LUT", linewidth=2)
#     plt.plot(x, x, '--', label="Identity LUT", alpha=0.5)
#     plt.xlabel("Input value (L)")
#     plt.ylabel("Enhanced value (L')")
#     title = f"1D LUT Curve"
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"✅ LUT 曲线图已保存到 {save_path}")
#
# def visualize_lut_1d(lut_param: torch.Tensor, save_path="lut_curve.png"):
#     """
#     可视化 1D LUT 学到的曲线，并与 identity 对比。
#
#     Args:
#         lut_param: [1, n_channels, n_bins] 的 torch.Tensor（LUT 参数）
#         save_path: 保存图像的路径
#     """
#     lut = lut_param.detach().cpu().numpy()[0]  # [n_channels, n_bins]
#     n_channels, n_bins = lut.shape
#     x = np.linspace(0, 1, n_bins)
#
#     plt.figure(figsize=(6, 5))
#     colors = ['r', 'g', 'b']
#     for i in range(n_channels):
#         plt.plot(x, lut[i], label=f"Channel {i} LUT", color=colors[i % 3], linewidth=2)
#
#     plt.plot(x, x, '--', label="Identity LUT", color='gray', alpha=0.5)
#     plt.xlabel("Input value (L)")
#     plt.ylabel("Enhanced value (L')")
#     plt.title("1D LUT Curve")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"✅ LUT 曲线图已保存到 {save_path}")
#
#     # 🔍 打印 LUT 数值
#     print("LUT 数值（每个通道）：")
#     for i, ch in enumerate(lut):
#         print(f"Channel {i}: {np.round(ch, 3)}")
# def visualize_lut_1d(lut_param: torch.Tensor, save_path="lut_curve.png"):
#     """
#     可视化 1D LUT 学到的曲线，并与 identity 对比（含 note 点、浅灰背景、RGB标注）。
#
#     Args:
#         lut_param: [1, 3, n_bins] 的 torch.Tensor（LUT 参数）
#         save_path: 保存图像的路径
#     """
#     lut = lut_param.detach().cpu().numpy()[0]  # [3, n_bins]
#     n_channels, n_bins = lut.shape
#     x = np.linspace(0, 1, n_bins)
#
#     plt.figure(figsize=(6, 5))
#     ax = plt.gca()
#
#     # ✅ 设置浅灰色背景
#     ax.set_facecolor('#f0f0f0')
#
#     # ✅ 通道标签和颜色
#     channel_names = ['Red', 'Green', 'Blue']
#     colors = ['r', 'g', 'b']
#
#     for i in range(n_channels):
#         plt.plot(x, lut[i], label=f"{channel_names[i]}",
#                  color=colors[i], linewidth=2, marker='o', markersize=4)
#
#     # ✅ identity LUT 对照线
#     plt.plot(x, x, '--', label="Identity", color='gray', alpha=0.6)
#
#     plt.xlabel("Input value (L)")
#     plt.ylabel("Enhanced value (L')")
#     plt.title("1D LUT Adjustment Curves")
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"✅ LUT 曲线图已保存到 {save_path}")
def visualize_lut_1d(lut_param: torch.Tensor, output_dir="output_lut_curves"):
    """
    可视化 1D LUT 学到的曲线，并与 identity 对比，每次保存到一个独立文件夹中。

    Args:
        lut_param: [1, 3, n_bins] 的 torch.Tensor（LUT 参数）
        output_dir: 主输出目录，内部将自动生成唯一子文件夹
    """
    # ✅ 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ✅ 创建唯一子目录（用时间戳或编号）
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 保存路径
    save_path = os.path.join(save_dir, "lut_curve.png")

    # ✅ 处理 LUT 数据
    lut = lut_param.detach().cpu().numpy()[0]  # [3, n_bins]
    n_channels, n_bins = lut.shape
    x = np.linspace(0, 1, n_bins)

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')

    channel_names = ['Red', 'Green', 'Blue']
    colors = ['r', 'g', 'b']

    for i in range(n_channels):
        plt.plot(x, lut[i], label=f"{channel_names[i]}",
                 color=colors[i], linewidth=2, marker='o', markersize=4)

    plt.plot(x, x, '--', label="Identity", color='gray', alpha=0.6)

    plt.xlabel("Input value (L)")
    plt.ylabel("Enhanced value (L')")
    plt.title("1D LUT Adjustment Curves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ LUT 曲线图已保存到：{save_path}")
    return save_dir  # ⬅️ 返回保存目录，便于存图像
#
# def plot_lut_vs_identity(lut, title="LUT vs Identity"):
#     """
#     lut: Tensor or numpy array of shape [3, N], where N is the LUT size (e.g., 17)
#     """
#     lut = lut.detach().cpu().numpy() if hasattr(lut, 'detach') else lut
#     N = lut.shape[1]
#     x = np.linspace(0, 1, N)
#
#     # Create identity array of shape [3, N], each channel's identity is the same as x
#     identity = np.tile(x, (3, 1))  # Shape: [3, N], each channel corresponds to identity curve
#
#     plt.figure(figsize=(8, 5))
#     for ch in range(3):
#         diff = lut[ch] - identity[ch]  # Now lut[ch] and identity[ch] have shape (17,)
#         plt.plot(x, diff, label=f'Channel {ch}')
#     plt.axhline(0, color='gray', linestyle='--')
#     plt.title(title)
#     plt.xlabel('Input Intensity')
#     plt.ylabel('LUT - Identity')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def save_gray_image(tensor, filename):
    """
    tensor: 2D tensor [H, W] or 3D [1, H, W]
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    img = TF.to_pil_image(tensor.clamp(0, 1))  # Normalize to [0,1]
    img.save(filename)
    print(f"Saved: {filename}")


def save_l_and_lut_out(L, L_lut_out, output_dir='vis_outputs'):
    """
    L, L_lut_out: Tensor of shape [B, 1, H, W]
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(L.shape[0], 4)):  # Save at most 4 samples
        save_gray_image(L[i], os.path.join(output_dir, f'{1}_L_{i}.png'))
        save_gray_image(L_lut_out[i], os.path.join(output_dir, f'{1}_L_lut_out_{i}.png'))
def visualize_multi_luts(luts, save_path_prefix="multi_lut_step"):
    """
    luts: tuple of (lut_dark, lut_mid, lut_bright), each [B, 3, N]
    Save separate plots for dark/mid/bright luts.
    """
    titles = ["Dark LUT", "Mid LUT", "Bright LUT"]
    for i, lut in enumerate(luts):
        plt.figure(figsize=(5, 4))
        for c, color in enumerate(['r', 'g', 'b']):
            plt.plot(torch.linspace(0, 1, lut.shape[-1]).cpu(), lut[0, c].detach().cpu().numpy(), color=color, label=f"{color.upper()}")
        plt.title(titles[i])
        plt.xlabel("Input Intensity")
        plt.ylabel("Output Intensity")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{save_path_prefix}_{titles[i].replace(' ', '_')}.png")
        plt.close()
# def visualize_areas_exposure_info(L, base_dark=0.25, base_mid=0.65, base_bright=0.8,
#                                   dark_threshold=0.15, bright_threshold=0.8,
#                                   dark_target_ref=0.3, bright_target_ref=0.2,
#                                   target_adjust_coeff=0.3,
#                                   max_dark_weight=0.4, max_bright_weight=0.3,
#                                   dark_loss_weight=3.0, bright_loss_weight=2.0):
#     """
#     打印区域曝光loss中各区域的重要指标，辅助调参与理解训练表现。
#     参数:
#         L: 输入照度图 (B, 1, H, W)，建议 B=1 单张图时使用
#     """
#
#     def brightness_regions(L, dark_threshold, bright_threshold):
#         dark_region = (L < dark_threshold)
#         bright_region = (L > bright_threshold)
#         mid_region = ~(dark_region | bright_region)
#         return dark_region, mid_region, bright_region
#
#     dark_region, mid_region, bright_region = brightness_regions(L, dark_threshold, bright_threshold)
#
#     eps = 1e-6
#     total = dark_region.sum() + mid_region.sum() + bright_region.sum() + eps
#
#     r_dark = dark_region.sum().float() / total
#     r_mid = mid_region.sum().float() / total
#     r_bright = bright_region.sum().float() / total
#
#     target_dark = base_dark + target_adjust_coeff * (dark_target_ref - r_dark).clamp(min=0)
#     target_mid = base_mid
#     target_bright = base_bright - target_adjust_coeff * (r_bright - bright_target_ref).clamp(min=0)
#
#     dark_mean = L[dark_region].mean().item() if dark_region.any() else 0
#     mid_mean = L[mid_region].mean().item() if mid_region.any() else 0
#     bright_mean = L[bright_region].mean().item() if bright_region.any() else 0
#
#     w_dark = r_dark.clamp(max=max_dark_weight)
#     w_mid = r_mid
#     w_bright = r_bright.clamp(max=max_bright_weight)
#
#     weight_sum = w_dark + w_mid + w_bright + eps
#     w_dark /= weight_sum
#     w_mid /= weight_sum
#     w_bright /= weight_sum
#
#     loss_dark = dark_loss_weight * w_dark * (dark_mean - target_dark) ** 2
#     loss_mid = w_mid * (mid_mean - target_mid) ** 2
#     loss_bright = bright_loss_weight * w_bright * (bright_mean - target_bright) ** 2
#
#     # 打印核心信息
#     print(f"[区域曝光信息] 💡")
#     print(f"区域比例 - 暗区: {r_dark:.3f}, 中区: {r_mid:.3f}, 亮区: {r_bright:.3f}")
#     print(f"亮度均值 - 暗区: {dark_mean:.3f}, 中区: {mid_mean:.3f}, 亮区: {bright_mean:.3f}")
#     print(f"目标亮度 - 暗区: {target_dark:.3f}, 中区: {target_mid:.3f}, 亮区: {target_bright:.3f}")
#     print(f"区域权重 - 暗区: {w_dark:.3f}, 中区: {w_mid:.3f}, 亮区: {w_bright:.3f}")
#     print(f"区域 loss - 暗区: {loss_dark:.4f}, 中区: {loss_mid:.4f}, 亮区: {loss_bright:.4f}")
#     print(f"总 Loss（区域曝光）= {loss_dark + loss_mid + loss_bright:.4f}")

class IAT(nn.Module):
    def __init__(self):
        super(IAT, self).__init__()
        # self.local_net = Local_pred()
        # if self.training:

        self.N_net = N_net(num=64)
        self.L_net = L_net(num=48)
        self.scien = EnhanceNetwork(layers=1,channels=4)
        self._criterion = LossFunction()
        self.R_net = R_net(num=64)
        # self.L_enhance_net = L_enhance_net(in_channel=1, num=32)
        self.L_enhance_net = LearnableLUT1D()
        self.illp = F_light_prior_estimater(n_fea_middle=64)
        self.outl2out = nn.Conv2d(3, 1, 1)

    def forward(self, input):
        # print(input.shape )
        # x = self.N_net(input)  # decome
        L = self.L_net(input)  # 照度分量
        # L,_=self.scien(input)
        r = input / L  # 粗反射
        r = torch.clamp(r, 0, 1)
        r_lut_out, lut1d = self.L_enhance_net(r)  # Eenhance#lut调整的反射分量结果
        # save_gray_image(L[0], "L_gray.png")  # 保存保存灰度图

        # y, illumap = self.illp(input)  # map和fea
        # r2 = self.R_net(input, y)  # 反射分量 输入照度信息

        # L2= torch.clamp(self.outl2out(L2),0,1)
        # print(L.shape)

        # 调整光照gamma？ r是反射分量

        # print(lut1d.shape)torch.Size([2, 3, 17])
        # 应用颜色校正

        # plot_lut_vs_identity(lut1d)

        # visualize_lut_1d(lut1d, save_path=f"lut_step_{1}.png") #一个
        # visualize_lut_1d(lut1d)  #保存时间的 一个返回路径
        # enhanced_image_path = os.path.join(save_dir, "enhanced_image.png")
        # save_image(lut1d, enhanced_image_path)  # 保存网络输出图像
        # save_l_and_lut_out(r, r_lut_out) #对比图
# # #
        # visualize_multi_luts(lut1d)#三个

        # return L, r

        # plot_rgb_luts(lut1d, save_path="rgb_lut.png")


        return L, r, lut1d, r_lut_out  # r2和rlut都可以做为结果


    def areas_exposure_loss(
            self,
            L,
            base_dark=0.15,  # ✅ 可适当增大，如 0.15 或 0.2，提高暗区亮度，增强对比
            base_mid=0.75,
            base_bright=0.7,  # ✅ 可适当减小，如 0.7 或 0.6，降低亮区亮度，防止过曝
            dark_threshold=0.3,
            bright_threshold=0.6,  # ✅ 可减小，如 0.7，使更多区域划为亮区，参与抑制
            dark_target_ref=0.3,
            bright_ratio_threshold=0.05,
            target_adjust_coeff=0.3,  # ✅ 可增大如 0.3，增强曝光目标的动态性，防止泛白
            max_dark_weight=0.15,  # ✅ 可适当减小如 0.15，避免暗区主导loss
            max_bright_weight=0.1,  # ✅ 可适当减小如 0.1，降低亮区过曝风险
            dark_loss_weight=2.0,
            bright_loss_weight=3.0  # ✅ 可增大如 3.0，加强亮区损失惩罚，抑制过亮
    ):
        # L = rout.mean(dim=1, keepdim=True)  # 灰度图
        """
        区域感知曝光损失，根据区域比例动态调整曝光目标和区域权重。
        参数:
            L: 输入图像的照度图 (B, 1, H, W)
        """

        # def rgb_to_y(rgb):
        #     return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]
        #
        # # L = Lout.mean(dim=1, keepdim=True)  # 形状变为 (B, 1, H, W)
        #
        # L = rgb_to_y(Lout)  # 也会变为 (B, 1, H, W)

        # 直方图zhifang
        #
        # L_np = L.detach().cpu().numpy().flatten()
        # plt.hist(L_np, bins=100, range=(0, 1))
        # plt.title("Illumination Histogram")
        # plt.xlabel("Luminance")
        # plt.ylabel("Pixel Count")
        # plt.show()


        def brightness_regions(L):
            """根据亮度阈值划分区域"""
            dark_region = (L < dark_threshold)
            bright_region = (L > bright_threshold)
            mid_region = ~( dark_region | bright_region)
            return dark_region, mid_region, bright_region

        dark_region, mid_region, bright_region = brightness_regions(L)

        eps = 1e-6
        total = dark_region.sum() + mid_region.sum() + bright_region.sum() + eps

        # 区域比例
        r_dark = dark_region.sum().float() / total
        r_mid = mid_region.sum().float() / total
        r_bright = bright_region.sum().float() / total

        # 动态曝光目标
        target_dark = base_dark + target_adjust_coeff * (dark_target_ref - r_dark).clamp(min=0)
        target_mid = base_mid
        target_bright = base_bright - target_adjust_coeff * (r_bright - bright_ratio_threshold).clamp(min=0)

        # 区域亮度均值
        dark_mean = L[dark_region].mean() if dark_region.any() else 0
        mid_mean = L[mid_region].mean() if mid_region.any() else 0
        bright_mean = L[bright_region].mean() if bright_region.any() else 0

        # 区域权重
        w_dark = r_dark.clamp(max=max_dark_weight)
        w_mid = r_mid
        w_bright = r_bright.clamp(max=max_bright_weight)

        weight_sum = w_dark + w_mid + w_bright + eps
        w_dark /= weight_sum
        w_mid /= weight_sum
        w_bright /= weight_sum

        # 加权损失
        loss = (
                dark_loss_weight * w_dark * (dark_mean - target_dark) ** 2 +
                w_mid * (mid_mean - target_mid) ** 2 +
                bright_loss_weight * w_bright * (bright_mean - target_bright) ** 2
        )
        print(f"mean dark: {dark_mean:.3f}, mid: {mid_mean:.3f}, bright: {bright_mean:.3f}")
        print(f"ratio dark: {r_dark:.3f}, mid: {r_mid:.3f}, bright: {r_bright:.3f}")
        return loss

    def dark_region_constraint(self,I_hat,L ,max_dark_value=0.3):
        gray = L # 灰度图
        dark_region = gray < 0.2
        return ((gray[dark_region] - max_dark_value) ** 2).mean()

    def color_loss(self, lut_out):
        mean_r = lut_out[:, 0, :, :].mean()
        mean_g = lut_out[:, 1, :, :].mean()
        mean_b = lut_out[:, 2, :, :].mean()
        return ((mean_r - mean_g) ** 2 + (mean_g - mean_b) ** 2 + (mean_b - mean_r) ** 2)

    # def color_consistency_loss(self,R):
    #     r, g, b = R[:, 0:1], R[:, 1:2], R[:, 2:3]
    #     return (torch.abs(r - g) + torch.abs(r - b) + torch.abs(g - b)).mean()
    def channel_balance_loss(self,img):
        means = torch.mean(img, dim=[2, 3])  # shape: [B, 3]
        mean_r, mean_g, mean_b = means[:, 0], means[:, 1], means[:, 2]
        rg = torch.abs(mean_r - mean_g)
        gb = torch.abs(mean_g - mean_b)
        rb = torch.abs(mean_r - mean_b)
        return (rg + gb + rb).mean()

    def lut_deviation_loss(self, lut):
        n_bins = lut.shape[-1]
        identity = torch.linspace(0, 1, n_bins, device=lut.device).view(1, 1, -1)
        return F.l1_loss(lut, identity.expand_as(lut))

        # r, g, b = lut[:, 0], lut[:, 1], lut[:, 2]
        # return ((r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2).mean()

    def phase_loss(self, r, lut_out):
        phase_input = torch.angle(torch.fft.fft2(r, norm='ortho'))
        phase_hat = torch.angle(torch.fft.fft2(lut_out, norm='ortho'))
        return F.l1_loss(phase_input, phase_hat)

    def lut_smoothness_loss(self, lut):
        diff = lut[:, :, 1:] - lut[:, :, :-1]
        return (diff ** 2).mean()

    def high_order_smoothness_loss(self, lut):
        second_diff = lut[:, :, 2:] - 2 * lut[:, :, 1:-1] + lut[:, :, :-2]
        return (second_diff ** 2).mean()

    def enhance_image_loss(self, r_lut_out, r, threshold=0.05):
        diff = r_lut_out - r
        enhance_loss = F.relu(threshold - diff).mean()
        return enhance_loss

    def L_mean_loss(self, L, target=0.6):
        # 控制亮度图 L 的平均值不偏暗不偏亮
        return ((L.mean(dim=[1, 2, 3]) - target) ** 2).mean()

        # 能变成正常曲线的

    def exposure_loss(self, I_hat, target_mean=0.65):
        gray = I_hat.mean(dim=1, keepdim=True)  # 灰度图
        patch_mean = F.avg_pool2d(gray, kernel_size=16)
        return ((patch_mean - target_mean) ** 2).mean()

    def tv_loss(self,illumination):
        def gradient(img):
            height = img.size(2)
            width = img.size(3)
            gradient_h = (img[:, :, 2:, :] - img[:, :, :height - 2, :]).abs()
            gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width - 2]).abs()
            return gradient_h, gradient_w
        gradient_illu_h, gradient_illu_w = gradient(illumination)

        loss_h = gradient_illu_h

        loss_w = gradient_illu_w

        loss = loss_h.mean() + loss_w.mean()

        return loss

    def selective_enhance_loss(self, r_lut_out, r):
        diff = r_lut_out - r
        low_mask = (r <= 0.1)
        mid_mask = (r > 0.2) & (r < 0.7)
        high_mask = (r > 0.9)

        loss_low = F.relu(diff[low_mask]).mean()  # 不增强暗区
        loss_mid = -diff[mid_mask].mean()  # 增强中间
        loss_high = F.relu(diff[high_mask] - 0.0).mean()  # 压制亮部
        return 1.0 * loss_low + 3.0 * loss_mid + 1.0 * loss_high

    def sciloss(self, input):
        L, r, lut1d, r_lut_out= self(input)
        # L, r = self(input)
        # lossr = F.l1_loss(r2, r)
        # lossfin2 = F.l1_loss(L_lut_out, r2)
        # lossen = F.l1_loss(r, r)
        # loss = loss_i + lossr+lossfin2
        exposure_weight = 10.0
        color_weight = 5
        phase_weight = 1.0
        smooth_weight = 10 # 1600
        deviation_weight = 10.0
        enhance_weight = 15
        recon_weight = 10
        # dark_mask = (L < 0.3).float()
        # weighted_loss = (base_loss * (1 + dark_mask * 3)).mean()
        # 在 loss 中对暗区加权

        # loss_i = recon_weight * self._criterion(input, L)
        mse_loss = torch.nn.MSELoss()
        max_rgb1 = torch.max(input,1)[0].unsqueeze(1)  # 获取RGB最大值作为光照估计
        loss_i = mse_loss(L, max_rgb1) + self.tv_loss(L)  #光照估计约束 + 平滑性
        # loss_i = mse_loss(L, max_rgb1) + F.l1_loss(L,r)
        # save_gray_image(L2[0], "./Lout/L_gray.png")
        loss_i = recon_weight * loss_i
        # loss_i = mse_loss(L, max_rgb1)

        # loss1 = mse_loss(L, L2)

        # L2=input / r_lut_out
        # L2 = L2.mean(dim=1, keepdim=True)  # 灰度图
        # print(max_rgb1.shape)

        loss_exp = exposure_weight * self.exposure_loss(r_lut_out)


        # loss_exp = exposure_weight * self.areas_exposure_loss(L)
        # visualize_areas_exposure_info(L) #可视化曝光损失
        loss_col = color_weight * self.color_loss(r_lut_out)
        loss_pha = phase_weight * self.phase_loss(r, r_lut_out)  # 有错误应该L为r
        # loss_smooth = smooth_weight * self.lut_smoothness_loss(lut1d)
        loss_smooth = smooth_weight * self.lut_smoothness_loss(lut1d)
        #

        # loss_channel = self.channel_balance_loss(r_lut_out)
    #改L为单通道，exposure的阈值数

        #
        # print(
            # f"[Step] Losses | exp: {loss_exp:.4f} col: {loss_col:.4f} pha: {loss_pha:.4f} smooth: {loss_smooth:.4f} dev: {loo_dark:.4f} enlut: {loo_con:.4f} ")
        return loss_i + loss_exp + loss_col + loss_pha + loss_smooth
        # return loss_i

writer = SummaryWriter('./runs')


class APBSN(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''

    def __init__(self, pd_a=5, pd_b=2, pd_pad=0, R3=False, R3_T=8, R3_p=0.16,
                 bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9
                 ):
        '''
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-imageas by PD process
            R3             : flag of 'Random Replacing Refinement'

            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        self.threshold = 160
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise NotImplementedError('bsn %s is not implemented' % bsn)

    def forward(self, img, illu_map, i, pd):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''

        if pd is None: pd = self.pd_a
        b, c, h, w = img.shape
        # maskill = (i >0.18).float()
        maskill = (i > i.mean()).float()
        # maskill = (i > 0.18).float()
        # save_gray_image(maskill[0], "maskxin.png") #保存保存灰度图
        # print('maskill', maskill.shape)
        # 保存=================================
        # illuimg = maskill.mul(255).byte()
        # illuimg=illuimg[0,:,:,:]
        # bgr_image = illuimg.cpu().numpy().transpose(1, 2, 0)
        # bImg = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # plt.imshow(bImg)
        # plt.axis('off')
        # plt.savefig('image3.png', bbox_inches='tight', pad_inches=0)
        # plt.show()
        # 保存====================================

        # maskill = (i > i.mean()).float()
        maskill1 = (i < i.max()).float()
        # pad images for PD process
        if h % pd != 0:
            img = F.pad(img, (0, 0, 0, pd - h % pd), mode='constant', value=0)
            maskill = F.pad(maskill, (0, 0, 0, pd - h % pd), mode='constant', value=0)
        if w % pd != 0:
            img = F.pad(img, (0, pd - w % pd, 0, 0), mode='constant', value=0)
            maskill = F.pad(maskill, (0, pd - w % pd, 0, 0), mode='constant', value=0)
        pd_img = util2.pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        maskill = util2.pixel_shuffle_down_sampling(maskill, f=pd, pad=self.pd_pad)
        # print('maskillu',maskill.shape)

        pd_img, random2seq = util2.randomArrangement(pd_img, pd)

        pd_img_denoised = self.bsn(pd_img, maskill)

        pd_img_denoised = util2.inverseRandomArrangement(pd_img_denoised, random2seq, pd)
        img_pd_bsn = util2.pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)

        # save_l_and_lut_out(img, img_pd_bsn) #对比图
        # print('pd_img', img_pd_bsn.shape)
        # #==============================================================================================================================
        #
        # b,c,h,w=img.shape
        # img_pd2_bsn = img
        # maskill2 = (i >0.18).float()
        # if h % 2 != 0:
        #     img_pd2_bsn = F.pad(img_pd2_bsn, (0, 0, 0, 2 - h % 2), mode='constant', value=0)
        #     maskill2 = F.pad(maskill2, (0, 0, 0, 2 - h % 2), mode='constant', value=0)
        # if w % 2 != 0:
        #     img_pd2_bsn = F.pad(img_pd2_bsn, (0, 2 - w % 2, 0, 0), mode='constant', value=0)
        #     maskill2 = F.pad(maskill2, (0, 2 - w % 2, 0, 0), mode='constant', value=0)
        # pd2_img = util2.pixel_shuffle_down_sampling(img_pd2_bsn, f=2, pad=self.pd_pad)
        # maskill2 = util2.pixel_shuffle_down_sampling(maskill2, f=2, pad=self.pd_pad)
        # pd2_img, random2seq = util2.randomArrangement(pd2_img, 2)
        # pd2_img_denoised = self.bsn(pd2_img, maskill2)
        # pd2_img_denoised = util2.inverseRandomArrangement(pd2_img_denoised, random2seq, 2)
        # img_pd2_bsn = util2.pixel_shuffle_up_sampling(pd2_img_denoised, f=2, pad=self.pd_pad)
        #
        #
        #
        # # img_pd2_bsn = forward_mpd(img_pd2_bsn, pd=2)
        #
        # # # ============== PD = 5 ====================
        # img_pd5_bsn = img
        # maskill5 = (i > i.mean()).float()
        # if h % 5 != 0:
        #     img_pd5_bsn = F.pad(img_pd5_bsn, (0, 0, 0, 5 - h % 5), mode='constant', value=0)
        #     maskill5 = F.pad(maskill5, (0, 0, 0, 5 - h % 5), mode='constant', value=0)
        # if w % 5 != 0:
        #     img_pd5_bsn = F.pad(img_pd5_bsn, (0,5 - w % 5, 0, 0), mode='constant', value=0)
        #     maskill5 = F.pad(maskill5, (0, 5 - w % 5, 0, 0), mode='constant', value=0)
        # pd5_img = util2.pixel_shuffle_down_sampling(img_pd5_bsn, f=5, pad=self.pd_pad)
        # maskill5 = util2.pixel_shuffle_down_sampling(maskill5, f=5, pad=self.pd_pad)
        # pd5_img, random5seq = util2.randomArrangement(pd5_img, 5)
        # pd5_img_denoised = self.bsn(pd5_img, maskill5)
        # pd5_img_denoised = util2.inverseRandomArrangement(pd5_img_denoised, random5seq, 5)
        # img_pd5_bsn = util2.pixel_shuffle_up_sampling(pd5_img_denoised, f=5, pad=self.pd_pad)
        # img_pd5_bsn = img_pd5_bsn[:, :, :h, :w]
        # # ============== FUSE 1 ====================
        # maskill1 = (i > i.max()).float()
        # img_pd1_bsn = self.bsn(img, maskill1)
        # img_pd_bsn=img_pd1_bsn
        # img_pd_bsn = torch.add(torch.mul(img_pd5_bsn, 0.7), torch.mul(img_pd1_bsn, 0.3))  # 鍘? 9锛?1
        #
        # # ============== FUSE 2 ====================
        # img_pd_bsn = torch.add(torch.mul(img_pd_bsn, 0.2), torch.mul(img_pd2_bsn, 0.8))
        #     == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

        if not self.R3:
            # print('no r3')
            return img_pd_bsn, maskill
        # return img_pd_bsn
        else:
            denoised = torch.empty(*(img.shape), self.R3_T, device=img.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(img)
                mask = indice < 0.6

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = img[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                # print(tmp_input.shape) torch.Size([1, 3, 404, 604])
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input, maskill1)
                    # print(denoised.shape) #(400, 600, 3)
                else:
                    denoised[..., t] = self.bsn(tmp_input, maskill1)[:, :, p:-p, p:-p]
            rimg = torch.mean(denoised, dim=-1)
            # print('yes r3apbsn')
            return rimg, maskill
            # pd_img_denoised = self.bsn(oimg, maskill)
            #
            # =======================迭代======================
            # rimg=self.bsn(img_pd_bsn,maskill)
            # r2img=self.bsn(rimg,maskill)
            # r3img = self.bsn(r2img, maskill)
            # r4img = self.bsn(r3img, maskill)
            # r5img = self.bsn(r4img, maskill)
            # r6img = self.bsn(r5img, maskill)
            # # r7img = self.bsn(r6img, maskill)
            # # r8img = self.bsn(r7img, maskill)
            # # r10img[...]=rimg, r2img, r3img, r4img, r5img, r6img, r7img, r8img
            # # r9img=torch.mean(r10img, dim=-1)
            # return r6img,maskill
    # =======================迭代======================


def tensor2np(t: torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    # t = t.detach()

    # gray
    if len(t.shape) == 2:
        return t.permute(1, 2, 0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.permute(1, 2, 0).numpy(), axis=2)
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.permute(0, 2, 3, 1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s' % (t.shape,))


def get_mean(self, tensor_shapes):
    means = []
    for i in tensor_shapes:
        a = i.mean()  # mean方法可以直接用于tensor类型数据的计算
        means.append(a)
    # print("获得平均值了")
    return means[0]


class R2RNet(nn.Module):  # 总的
    def __init__(self):
        super(R2RNet, self).__init__()
        # self.bsn=DBSNl(in_ch=3, out_ch=3,R3=True)

        # self.DecomNet = DecomNet()#分解 需要vgg感知损失
        # self.DenoiseNet = DenoiseNet()#去噪cnn
        # self.RelightNet = RelightNet() #增强
        self.DecomNet = IAT()  # 分解 需要vgg感知损失

        self.DenoiseNet = APBSN()  # 去噪cnn
        self.illma = nn.Conv2d(3, 1, 1)
        self.lview = nn.Conv2d(1, 3, 1)
        # self.RelightNet = RelightNet()  # 增强
        # self.vgg = load_vgg16("./model")

    def exposure_loss(self, I_hat, target_mean=0.65):
        gray = I_hat.mean(dim=1, keepdim=True)  # 灰度图
        patch_mean = F.avg_pool2d(gray, kernel_size=16)
        return ((patch_mean - target_mean) ** 2).mean()

    def forward(self, input_low):
        # print(input_low.shape)
        # device = torch.device('cuda:0')
        # inputs = inputs
        # input_low1=torch.tensor(input_low)
        # input_low = tensor2np(input_low)
        # input_low = Variable(torch.FloatTensor(input_low1))

        # input_low = torch.FloatTensor(torch.tensor(input_low))
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        # input_low = Variable(input_low, requires_grad=False)
        # Forward DecomNet

        L, r, lut1d, r_lut_out= self.DecomNet(input_low)
        end = time.time()
        # inference_time = end - start
        # print(f"Inference Time: {inference_time *1e3:.4f} ms")
        # L, r= self.DecomNet(input_low)
        self.out1 = r_lut_out.detach().cpu()
        # self.out1 = r.detach().cpu()
        self.itrloss1 = self.DecomNet.sciloss(input_low)

        self.loss_Decom = self.itrloss1
        # self.out1 = 0.4*inlist[2].detach().cpu() + 0.6*input_low.detach().cpu()
        # self.out1 = 0.2 * rlist[2].detach().cpu() + 0.8 * input_low.detach().cpu()
        # self.out1 = attlist[2].detach().cpu()+input_low.detach().cpu()+ilist[2].detach().cpu()
        # self.out1= 1.2*rlist[2].detach().cpu()+ilist[2].detach().cpu()
        # self.out1= input_low.detach().cpu()*L.detach().cpu()+input_low.detach().cpu()


        # 去噪denoiseloss
        # # illu_map = self.illma(L)
        # #
        illu_map=L
        # print(L.shape)
        denoise_R1, maskill = self.DenoiseNet(r_lut_out, illu_map, illu_map, pd=4)
        denoise_R2, maskill = self.DenoiseNet(r_lut_out, illu_map, illu_map, pd=1)

        self.out3 = denoise_R2.detach().cpu()

        # 加权
        # lambda_bri = torch.nn.Parameter(torch.tensor(1.0))  # 初始值为1
        # lambda_dar = torch.nn.Parameter(torch.tensor(1.0))
        # lambda_deno = torch.nn.Parameter(torch.tensor(1.0))

        # self.denoise_loss = F.l1_loss(denoise_R1, r_lut_out) +  F.l1_loss(maskill * denoise_R1, maskill * r_lut_out)
        # print('lossmask',maskill.shape)
        # print('denoise_R1', denoise_R1.shape)
        # print('r_lut_out', r_lut_out.shape)
        # self.denoise_loss = lambda_deno*F.l1_loss(denoise_R1, r_lut_out) + \
        #                     lambda_bri*F.l1_loss(maskill * denoise_R1, maskill * r_lut_out) +  \
        #                     lambda_dar*F.l1_loss((1 - maskill) * denoise_R1, (1 - maskill) * r_lut_out)

        # self.denoise_loss2 =self.exposure_loss(denoise_R1)

        # self.denoise_loss =F.l1_loss(maskill*denoise_R1, maskill*r_lut_out)+lambda_dar*F.l1_loss((1-maskill)*denoise_R1, (1-maskill)*r_lut_out)
        # self.loss_Denoise = self.denoise_loss1
        # 原始
        self.denoise_loss = F.l1_loss(denoise_R1, r_lut_out)
        self.loss_Denoise = self.denoise_loss






    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        # self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))

        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, eval_high_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        spsnr = 0
        sssim = 0
        sdists = 0
        count = 0
        slpips = 0
        self.logger = Logger()
        # self.logger = Logger((33, 70191), log_dir=self.file_manager.get_dir(''),log_file_option='a')
        with torch.no_grad():  # Otherwise the intermediate gradient would take up huge amount of CUDA memory
            for idx in range(len(eval_low_data_names)):
                eval_low_img = Image.open(eval_low_data_names[idx])
                eval_low_img = np.array(eval_low_img, dtype="float32") / 255.0
                eval_low_img = np.transpose(eval_low_img, [2, 0, 1])

                input_low_eval = np.expand_dims(eval_low_img, axis=0)
                eval_high_img = Image.open(eval_high_data_names[idx])
                eval_high_img = np.array(eval_high_img, dtype="float32")

                if train_phase == "Decom":
                    self.forward(input_low_eval)  # 输入都是低质？
                    # result_1 = self.color
                    # result_2 = self.output_I_low
                    # input = np.squeeze(input_low_eval)
                    # result_1 = np.squeeze(result_1)
                    # result_2 = np.squeeze(result_2)
                    dcat_image = self.out1
                    dcat_image = dcat_image.numpy().squeeze(0)
                    cat_image = dcat_image
                    # deval_high_img = self.tar1
                    # deval_high_img = deval_high_img.numpy().squeeze(0)
                    # eval_high_img = deval_high_img
                    # cat_image = np.concatenate([result_1, result_2], axis=2) #输出是两个分解的cat
                if train_phase == 'Denoise':
                    self.forward(input_low_eval)
                    # result_1 = self.outdenoise
                    # input = np.squeeze(input_low_eval)
                    # denoise_R2,maskill = self.DenoiseNet(self.input2,pd=2)
                    # denoise_R2 = self.DenoiseNet.bsn(self.input2)
                    # denoise_R2 = self.DenoiseNet(self.input2, pd=1)
                    # self.out3 = denoise_R2.detach().cpu()
                    # x=self.input2
                    # denoised = torch.empty(*(x.shape), 8, device=x.device)
                    # for t in range(8):
                    #     indice = torch.rand_like(x)
                    #     mask = indice <0.16
                    #
                    #     tmp_input = torch.clone(denoise_R2).detach()
                    #     tmp_input[mask] = x[mask]
                    #     p = 2
                    #     tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                    #     # if 2 == 0:
                    #     #     denoised[..., t] = self.bsn(tmp_input, is_masked=True)
                    #     # else:
                    #     denoised[..., t] = self.bsn(tmp_input,maskill)[:, :, p:-p, p:-p]#加入maskill
                    #
                    # rimg_pd_bsn = torch.mean(denoised, dim=-1)

                    # x = self.input2
                    # denoised = torch.empty(*(x.shape), 8, device=torch.device('cuda:0'))
                    # for t in range(8):
                    #     indice = torch.rand_like(x)
                    #     mask = indice < 0.16
                    #     tmp_input = torch.clone(denoise_R2).detach()
                    #     tmp_input[mask] = x[mask]
                    #     p = 2
                    #     tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                    #     denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
                    # rimg_pd_bsn = torch.mean(denoised, dim=-1)
                    # self.out3 = rimg_pd_bsn.detach().cpu()
                    dcat_image = self.out3

                    dcat_image = dcat_image.numpy().squeeze(0)
                    cat_image = dcat_image
                    # deval_high_img = self.tar2
                    # deval_high_img=deval_high_img.numpy().squeeze(0)
                    # eval_high_img = deval_high_img
                # if train_phase == "Relight":
                #     self.forward(input_low_eval, input_low_eval)
                #     result_4 = self.output_S
                #     input = np.squeeze(input_low_eval)
                #     result_4 = result_4.numpy().squeeze(0)
                #     cat_image = result_4

                cat_image = np.transpose(cat_image, (1, 2, 0))
                # print(cat_image.shape)
                im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
                im_test = np.array(im, dtype='float32')
                # im_eval =eval_high_img
                # im_eval= Image.fromarray(np.clip(eval_high_img * 255.0, 0, 255.0).astype('uint8'))
                # im_eval=np.array(im_eval, dtype='float32')
                spsnr += psnr2(im_test, eval_high_img)
                sssim += ssim2(im_test, eval_high_img)
                slpips += lpips2(im_test, eval_high_img)
            print('psnr=', spsnr / len(eval_low_data_names))
            print('ssim=', sssim / len(eval_low_data_names))
            print('lpips=', slpips / len(eval_low_data_names))
            writer.add_scalars('runs/metrics', {
                'psnr': spsnr / len(eval_low_data_names),
                'ssim': sssim / len(eval_low_data_names),
                'lpips': slpips / len(eval_low_data_names)
            }, epoch_num)

            self.logger.val('[%s] Done! PSNR : %.3f dB, SSIM : %.4f, LPIPS : %.4f' % (
                epoch_num, spsnr / len(eval_low_data_names), sssim / len(eval_low_data_names),
                slpips / len(eval_low_data_names)))

    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name = save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        if self.train_phase == 'Denoise':
            torch.save(self.DenoiseNet.state_dict(), save_name)
        # if self.train_phase == 'Relight':
        #     torch.save(self.RelightNet.state_dict(), save_name)

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + '/' + self.train_phase + '/'
        # load_dir = ckpt_dir
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts) > 0:
                load_ckpt = load_ckpts[-1]
                global_step = int(load_ckpt[:-4])
                ckpt_dict = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                if self.train_phase == 'Denoise':
                    self.DenoiseNet.load_state_dict(ckpt_dict)
                # if self.train_phase == 'Relight':
                #     self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0

    def train(self,
              train_low_data_names,
              eval_low_data_names,
              eval_high_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):

        numBatch = len(train_low_data_names) // int(batch_size)
        self.patch_size = patch_size
        # Create the optimizers
        self.train_op_Decom = optim.Adam(self.DecomNet.parameters(),
                                         lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)
        self.train_op_Denoise = optim.Adam(self.DenoiseNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)
        # self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
        #                                    lr=lr[0], betas=(0.9, 0.999), weight_decay=0.0001)

        # Initialize a network if its checkpoint is available
        self.train_phase = train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
              (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Denoise.param_groups:
                param_group['lr'] = self.lr
            # for param_group in self.train_op_Relight.param_groups:
            #     param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                # batch_input_high = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32') / 255.0
                    # train_high_img = Image.open(train_high_data_names[image_id])
                    # train_high_img = np.array(train_high_img, dtype='float32')/255.0
                    # Take random crops
                    h, w, _ = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    # train_high_img = train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        # train_high_img = np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        # train_high_img = np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        # train_high_img = np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    # train_high_img = np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    # batch_input_high[patch_id, :, :, :] = train_high_img
                    self.input_low = batch_input_low
                    # self.input_high = batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    # if image_id == 0:
                    # tmp = list(zip(train_low_data_names, train_high_data_names))
                    # random.shuffle(list(tmp))
                    # train_low_data_names, train_high_data_names = zip(*tmp)

                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    global_step += 1
                    loss = self.loss_Decom.item()
                elif self.train_phase == 'Denoise':
                    self.train_op_Denoise.zero_grad()
                    self.loss_Denoise.backward()
                    self.train_op_Denoise.step()
                    global_step += 1
                    loss = self.loss_Denoise.item()
                # elif self.train_phase == "Relight":
                #     self.train_op_Relight.zero_grad()
                #     self.loss_Relight.backward()
                #     self.train_op_Relight.step()
                #     global_step += 1
                #     loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_scalar('runs/loss', loss, global_step)
                img = torch.rand(3, 3, self.patch_size, self.patch_size).numpy()
                if global_step % 10 == 0:
                    img[:1, :, :, :] = batch_input_low[:1, :, :, :]
                    img[1:2, :, :, :] = self.out1[:1, :, :, :]
                    # img[2:3, :, :, :] = batch_input_high[:1, :, :, :]
                    writer.add_images('results', img)

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.save(iter_num, ckpt_dir)
                self.evaluate(epoch + 1, eval_low_data_names, eval_high_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)

        print("Finished training for phase %s." % train_phase)

    def predict(self,
                test_low_data_names,
                res_dir1,
                res_dir2,
                ckpt_dir,
                eval_high_data_names):

        # Load the network with a pre-trained checkpoint
        self.train_phase = 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        print(load_model_status)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
            self.DecomNet.eval()
        else:
            print("No pretrained model to restore!")
            raise Exception

        self.train_phase = 'Denoise'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
            self.DenoiseNet.eval()  #
        else:
            print("No pretrained model to restore!")
            raise Exception

        # self.train_phase = 'Relight'
        # load_model_status, _ = self.load(ckpt_dir)
        # if load_model_status:
        #     print(self.train_phase, ": Model restore success!")
        # else:
        #     print("No pretrained model to restore!")
        #     raise Exception

        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False
        # ssim = SSIM()
        # psnr = PSNR()
        ssim1_list = []
        psnr1_list = []
        lpips1_list = []
        psnr1_value = 0
        ssim1_value = 0
        lpips1_value = 0
        psnr2_value = 0
        ssim2_value = 0
        lpips2_value = 0
        count = 0
        # psnr1_list = []
        ssim2_list = []
        psnr2_list = []
        lpips2_list = []
        # psnr2_list = []
        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            count += 1
            test_img_path = test_low_data_names[idx]
            test_img_name = test_img_path.split('/')[-1]

            print('Processing ', test_img_name)
            test_low_img = Image.open(test_img_path)
            test_low_img = np.array(test_low_img, dtype="float32") / 255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            self.forward(input_low_test)
            result_1 = self.out1
            result_2 = self.out3
            # result_3 = self.output_I_delta
            # result_4 = self.output_S
            input = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            # result_3 = np.squeeze(result_3)
            # result_4 = np.squeeze(result_4)
            if save_R_L:
                cat1_image = np.concatenate([input, result_1], axis=2)
            else:
                cat1_image = result_1.numpy()
                cat2_image = result_2.numpy()
            cat1_image = np.transpose(cat1_image, (1, 2, 0))
            # cat1_image1=np.array(cat1_image, dtype='float32')
            cat2_image = np.transpose(cat2_image, (1, 2, 0))
            # print(cat2_image.shape)
            # cat2_image = cat2_image
            # print(cat_image.shape)
            im1 = Image.fromarray(np.clip(cat1_image * 255.0, 0, 255.0).astype('uint8'))
            # im11 = np.array(im1, dtype='float32')
            filepath = res_dir1 + '/' + test_img_name
            # im1.save(filepath[:-4] + 'illu' + '.jpg')
            im1.save(filepath[:-4] + 'illu' + '.png')
            # im1.save(filepath[:-4] + 'illu' + '.bmp')
            eval_high_img = Image.open(eval_high_data_names[idx])
            eval_high_img = np.array(eval_high_img, dtype="float32")
            # print('im1',cat1_image.device)
            # print('high',eval_                high_img.device)
            im11 = Image.fromarray(np.clip(cat1_image * 255.0, 0, 255.0).astype('uint8'))
            im_test1 = np.array(im11, dtype='float32')

            score_ssim = ssim2(im_test1, eval_high_img)
            score_psnr = psnr2(im_test1, eval_high_img)
            score_lpips = lpips2(im_test1, eval_high_img)

            ssim1_value += score_ssim
            psnr1_value += score_psnr
            lpips1_value += score_lpips
            # print('单幅图增强的 PSNR1 Value is:', psnr1_value)
            # print('单幅图增强的 SSIM1 Value is:', ssim1_value)
            # print('单幅图增强的 lpips1 Value is:', lpips1_value)
            im2 = Image.fromarray(np.clip(cat2_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = res_dir2 + '/' + test_img_name
            # im2.save(filepath[:-4] + 'deno' + '.jpg')
            im2.save(filepath[:-4] + 'deno' + '.png')
            # im2.save(filepath[:-4] + 'deno' + '.bmp')
            # im22 = Image.fromarray(np.clip(cat2_image * 255.0, 0, 255.0).astype('uint8'))
            im_test2 = np.array(im2, dtype='float32')
            ssim2_value += ssim2(im_test2, eval_high_img)
            psnr2_value += psnr2(im_test2, eval_high_img)
            lpips2_value += lpips2(im_test2, eval_high_img)
            # count+=count
        print('psnr1=', psnr1_value / count)
        print('ssim1=', ssim1_value / len(test_low_data_names))
        print('lpips1=', lpips1_value / len(test_low_data_names))
        print('ssim2=', ssim2_value / len(test_low_data_names))
        print('psnr2=', psnr2_value / len(test_low_data_names))
        print('lpips2=', lpips2_value / len(test_low_data_names))

        # ssim2_value = ssim2(im2, eval_high_img)
        # psnr2_value = psnr2(im_test2, eval_high_img)
        # lpips2_value = lpips2(im_test2, eval_high_img)
        # print('单幅图去噪的 PSNR2 Value is:', psnr2_value)
        # print('单幅图增强的 SSIM2 Value is:', ssim2_value)
        # print('单幅图增强的 lpips2 Value is:', lpips2_value)

        #     ssim1_list.append(ssim1_value)
        #     lpips1_list.append(lpips1_value)
        #     psnr1_list.append(psnr1_value)
        #     ssim2_list.append(ssim2_value)
        #     psnr2_list.append(psnr2_value)
        #     lpips2_list.append(lpips2_value)
        # SSIM1_mean = np.mean(ssim1_list)
        # PSNR1_mean = np.mean(psnr1_list)
        # lpips1_mean = np.mean(lpips1_list)
        # print('照度总的 SSIM1 Value is:', SSIM1_mean)
        # print('照度总的 PSNR1 Value is:', PSNR1_mean)
        # print('照度总的 lpips1 Value is:', lpips1_mean)
        # SSIM2_mean = np.mean(ssim2_list)
        # PSNR2_mean = np.mean(psnr2_list)
        # lpips2_mean = np.mean(lpips2_list)
        # print('去噪总的 SSIM2 Value is:', SSIM2_mean)
        # print('去噪总的 PSNR2 Value is:', PSNR2_mean)
        # print('去噪总的 lpips2 Value is:', lpips2_mean)