import torch
import json
import os

from torch import nn
from pathlib import Path
from torchsummary import summary

class Block(nn.Module):
    def __init__(self, inp_channels = 64, out_channels = 64, is_down = False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(inp_channels,
                               out_channels,
                               kernel_size = 3,
                               stride = 2 if is_down else 1,
                               padding = 1,
                               bias = False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.is_down = is_down
        self.down = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x0 = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.is_down:
            x0 = self.down(x0)
            
        x = x + x0
        x = self.relu(x)
        
        return x


class Resnet(nn.Module):
    def __init__(self, 
                 inp_channels, 
                 num_residual_block = [2, 2, 2, 2], 
                 num_class = 1,
                 vector = False):
        super().__init__()
        
        self.vector = vector
        self.first_channels = 64
        self.conv1 = nn.Conv2d(inp_channels,
                               out_channels = self.first_channels,
                               kernel_size = 7,
                               stride = 2,
                               padding = 3,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(self.first_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride = 2,
                                    padding = 1,)
        self.resnet, out_channels = self.resnet_layer(num_residual_block)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features = out_channels,
                            out_features = num_class,
                            bias = True)
        self.sm = nn.Softmax(dim=1)
    
    def resnet_layer(self, num_residual_block):
        resnet = []

        inp_channels = out_channels = self.first_channels
        is_down = False
        for numBlock in num_residual_block:
            for _ in range(numBlock):
                resnet.append(Block(inp_channels, out_channels, is_down))
                inp_channels = out_channels
                is_down = False
            is_down = True
            out_channels = inp_channels * 2
        
        return nn.Sequential(*resnet), out_channels // 2
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        if self.vector:
            return x
        x = self.fc(x)
        x = self.sm(x)
        return x



def create_model(out_dir = None,
                 model = Resnet,
                 **kwargs):
    
    if out_dir is not None:
        Path(os.path.join(out_dir, 'config')).mkdir(parents=True, exist_ok=True)
        save_kwargs = dict(locals())
        save_kwargs['model'] = model.__name__
        print(save_kwargs)
        json.dump(save_kwargs, open(os.path.join(out_dir, 'config', 'Model.json'), 'w'))
        print('[INFO]: Model config saved!')
        
    return model(**kwargs)