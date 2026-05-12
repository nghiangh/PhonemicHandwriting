import torch
from torch import nn

from PIL import Image
from torchvision import transforms, models

import os
import numpy as np
from tqdm import tqdm

def resize_keep_ratio(img):

    target_height = 64
    w, h = img.size
    new_width = int(w * (target_height / h))

    return img.resize((new_width, target_height))

transform = transforms.Compose([
    transforms.Lambda(resize_keep_ratio),
    transforms.ToTensor(),
])

class ResnetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

model = ResnetBackbone().cuda()

BASE_DIR = "UIT_HWDB_line"
IMAGE_DIR = "images"
FEATURE_DIR = "features"
split = "test"

if not os.path.isdir(os.path.join(BASE_DIR, FEATURE_DIR, split)):
    os.makedirs(os.path.join(BASE_DIR, FEATURE_DIR, split))

for img_file in tqdm(os.listdir(os.path.join(BASE_DIR, IMAGE_DIR, split))):
    image = Image.open(os.path.join(BASE_DIR, IMAGE_DIR, split, img_file)).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).cuda()
    feature = model(image)
    _, D, H, W = feature.shape
    feature = feature.reshape(D, H*W).transpose(0, 1)
    feature = feature.cpu().detach().numpy()

    img_id = img_file.split(".")[0]

    np.save(os.path.join(BASE_DIR, FEATURE_DIR, split, f"{img_id}.npy"), feature, allow_pickle=False)
