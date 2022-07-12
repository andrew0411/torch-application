'''
[reference] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
'''

import torch
import torch.nn as nn
from torchvision.models import resnet

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101']

def conv3x3(in_channel, out_channel, kernel_size=3, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm=None):
        super(BasicBlock, self).__init__()
        norm = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channel, out_channel, kernel_size=3, stride=stride)
        self.bn1 = norm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = norm(out_channel)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    '''
    Torchvision에서는 downsampling을 3x3 convolution에서 진행하고,
    원래 논문 [Deep residual learning for image recognition]은 1x1 convolution에서 stride 수행
    나도 3x3 convolution에서 stride
    
    official code애서는 mid_channel 대신 width 사용하고 최종 output이 expansion x out_channel 인데 덜 직관적임
    '''
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm=None):
        super(Bottleneck, self).__init__()
        norm = nn.BatchNorm2d

        mid_channel = out_channel // self.expansion
        self.conv1 = conv1x1(in_channel, mid_channel)
        self.bn1 = norm(mid_channel)
        self.conv2 = conv3x3(mid_channel, mid_channel, stride)
        self.bn2 = norm(mid_channel)
        self.conv3 = conv1x1(mid_channel, out_channel)
        self.bn3 = norm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    
        