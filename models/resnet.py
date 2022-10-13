import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding,
                     bias=False)


def conv1x1(inplanes, out_channels, stride=1):
    return nn.Conv2d(in_channels=inplanes, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.stride_look = stride
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout_1 = nn.Dropout(0.2)
        
        self.conv2 = conv3x3(in_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout_2 = nn.Dropout(0.2)


        self.residule = nn.Sequential()
        # needs 1x1 conv to convert dimensions
        if stride != 1 or in_channels != out_channels:    
            self.residule = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.2)
            )

    def forward(self, X):
        # save for residual connection
        out = F.relu(self.dropout_1(self.bn1(self.conv1(X))))
        # relu layer
        out = self.dropout_2(self.bn2(self.conv2(out)))
        out += self.residule(X)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.skip = nn.Sequential()

        if stride!=1 or inplanes!=self.expansion*planes:
            self.skip = nn.Sequential(
                conv1x1(inplanes, self.expansion*planes, stride),
                nn.BatchNorm2d(self.expansion*planes) )

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print(out.shape)
        identity = self.skip(x)
        # print(identity.shape)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion*planes, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.skip = nn.Sequential()
        if stride!=1 or inplanes != self.expansion*planes:
            self.skip = nn.Sequential(
                conv1x1(inplanes, self.expansion*planes, stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        identity = self.skip(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = block.expansion*planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def ResNet18(classes):
    return ResNet(ResidualBlock, [2,2,2,2], num_classes=classes)


def ResNet50(classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=classes)


def ResNet101(classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=classes)