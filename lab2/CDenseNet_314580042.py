from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn

class LDB(nn.Module):
    def __init__(self, in_channels, t = 0.5):
        super().__init__()
        out_channels = int(round(in_channels * t))        

        self.initial = nn.Conv2d(in_channels = in_channels,  out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        
        self.layer1 = nn.Sequential(
                            nn.BatchNorm2d(num_features = out_channels), 
                            nn.ReLU(inplace = True), 
                            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        )
        
        self.layer2 = nn.Sequential(
                            nn.BatchNorm2d(num_features = out_channels), 
                            nn.ReLU(inplace = True), 
                            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        )
        
        self.layer3 = nn.Sequential(
                            nn.BatchNorm2d(num_features = out_channels), 
                            nn.ReLU(inplace = True), 
                            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        )
        
        self.layer4 = nn.Sequential(
                            nn.BatchNorm2d(num_features = out_channels), 
                            nn.ReLU(inplace = True), 
                            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        )
        
        self.out_channels = in_channels + 4 * out_channels
    
    def forward(self, x):
        initial_output = self.initial(x)
        layer1_output = self.layer1(initial_output)
        layer2_output = self.layer2(layer1_output)

        layer3_input = layer1_output + layer2_output

        layer3_output = self.layer3(layer3_input)

        layer4_input = layer1_output + layer2_output + layer3_output

        layer4_output = self.layer4(layer4_input)
        
        return torch.cat([x, layer1_output, layer2_output, layer3_output, layer4_output], dim = 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = 32
        
        self.transition = nn.Sequential(
                            nn.Conv2d(in_channels = in_channels,  out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
                            nn.BatchNorm2d(num_features = out_channels),
                            nn.ReLU(inplace = True), 
        )

    def forward(self, x):
        return self.transition(x)

class CDenseNet(nn.Module):
    def __init__(self, n = 16, t = 0.5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 1,  out_channels = 32, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(inplace = True),
        )

        features = []
        for idx in range(n):
            ldb = LDB(in_channels = 32, t = t)
            features.append((f"ldb{idx + 1}", ldb))
            trans = TransitionLayer(in_channels = ldb.out_channels)
            features.append((f"transition{idx + 1}", trans))
        self.features = nn.Sequential(OrderedDict(features))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(in_features = 32, out_features = 128), 
            nn.ReLU(inplace = True), 
            nn.Linear(in_features = 128, out_features = 3)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x