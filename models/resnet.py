import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride_look = stride
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout_1 = nn.Dropout(0.1)
        
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout_2 = nn.Dropout(0.1)

        self.residule = nn.Sequential()
        if stride != 1 or in_channels != out_channels:    
            self.residule = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.1)
            )

    def forward(self, X):
        out = F.relu(self.dropout_1(self.bn1(self.conv1(X))))
        out = self.dropout_2(self.bn2(self.conv2(out)))
        out += self.residule(X)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, residual_block, num_classes):
        super().__init__()
        self.in_channel = 64
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(residual_block, 64, 2, stride=1)
        self.layer2 = self.make_layer(residual_block, 128, 2, stride=2)
        self.layer3 = self.make_layer(residual_block, 256, 2, stride=2)
        self.layer4 = self.make_layer(residual_block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_set in strides:
            layers.append(block(self.in_channel, channels, stride_set))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18(num_classes):
    return ResNet(residual_block, num_classes)