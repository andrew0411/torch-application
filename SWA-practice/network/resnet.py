'''
[reference] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
'''

from numpy import isin
import torch
import torch.nn as nn
from torchvision.models import resnet
import os

__all__ = ['ResNet18', 'ResNet50', 'ResNet101']

def conv3x3(in_channel, out_channel, kernel_size=3, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
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

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
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
    def __init__(self, block, layers, num_classes, zero_init_residual=False):
        super(ResNet, self).__init__()
        norm = nn.BatchNorm2d

        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 각 residual branch의 마지막 BN을 zero-initialize
        # 그래야지 residual branch가 0으로 시작해서 각 residual block이 identity function 처럼 작동
        # 이를 통해 0.2~0.3% 정도의 성능 향상
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)
    
    def make_layer(self, block, out_channel, n_blocks, stride):
        norm = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out_channel, stride),
                norm(out_channel)
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel
        
        for _ in range(1, n_blocks):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, shape, num_classes, checkpoint_dir='checkpoint', checkpoint_name='ResNet18',
                 pretrained=False, pretrained_path=None, zero_init_residual=False):
        super(ResNet18, self).__init__()

        if len(shape) != 3:
            raise ValueError('Invalid shape: {}.format(shape')

        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.H, self.W, self.C = shape

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, 'model.pt')

        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.num_classes, zero_init_residual=zero_init_residual)

        if pretrained:
            print('Loading pretrained weight')
            model = resnet.resnet18(pretrained=True)
            if zero_init_residual:
                for m in model.modules():
                    if isinstance(m, resnet.Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
        
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_features = 512 * BasicBlock.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes)
        
    def load(self, checkpoint_name=''):
        if checkpoint_name == '':
            self.load_state_dict(torch.load(self.checkpoint_path))
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + 'pt')
            self.load_state_dict(torch.load(checkpoint_path))

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + 'pt')
            torch.save(self.state_dict(), self.checkpoint_path)
    
    def forward(self, x):
        out = x
        out = self.features(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, shape, num_classes, checkpoint_dir='checkpoint', checkpoint_name='ResNet18',
                 pretrained=False, pretrained_path=None, zero_init_residual=False):
        super(ResNet18, self).__init__()

        if len(shape) != 3:
            raise ValueError('Invalid shape: {}.format(shape')

        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.H, self.W, self.C = shape
        

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, 'model.pt')

        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=self.num_classes, zero_init_residual=zero_init_residual)

        if pretrained:
            print('Loading pretrained weight')
            model = resnet.resnet50(pretrained=True)
            if zero_init_residual:
                for m in model.modules():
                    if isinstance(m, resnet.Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
        
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_features = 512 * BasicBlock.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes)
        
    def load(self, checkpoint_name=''):
        if checkpoint_name == '':
            self.load_state_dict(torch.load(self.checkpoint_path))
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + 'pt')
            self.load_state_dict(torch.load(checkpoint_path))

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + 'pt')
            torch.save(self.state_dict(), self.checkpoint_path)
    
    def forward(self, x):
        out = x
        out = self.features(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class ResNet101(nn.Module):
    def __init__(self, shape, num_classes, checkpoint_dir='checkpoint', checkpoint_name='ResNet18',
                 pretrained=False, pretrained_path=None, zero_init_residual=False):
        super(ResNet18, self).__init__()

        if len(shape) != 3:
            raise ValueError('Invalid shape: {}.format(shape')

        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.H, self.W, self.C = shape

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, 'model.pt')

        model = ResNet(BasicBlock, [3, 4, 23, 3], num_classes=self.num_classes, zero_init_residual=zero_init_residual)

        if pretrained:
            print('Loading pretrained weight')
            model = resnet.resnet101(pretrained=True)
            if zero_init_residual:
                for m in model.modules():
                    if isinstance(m, resnet.Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
        
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_features = 512 * BasicBlock.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes)
        
    def load(self, checkpoint_name=''):
        if checkpoint_name == '':
            self.load_state_dict(torch.load(self.checkpoint_path))
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + 'pt')
            self.load_state_dict(torch.load(checkpoint_path))

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name, checkpoint_name + 'pt')
            torch.save(self.state_dict(), self.checkpoint_path)
    
    def forward(self, x):
        out = x
        out = self.features(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out



