'''
CIFAR10 다운받아서 SWA로 학습해보기
'''

import os
import torch
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from PIL import Image, ImageFile

# 잘린 이미지가 있어도 에러 발생 안 함
ImageFile.LOAD_TRUNCATED_IMAGES = True

