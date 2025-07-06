import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from torch.autograd import Function


# Кастомные слои

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        out = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        return torch.tanh(out)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attn = torch.bmm(q.permute(0, 2, 1), k)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class CustomActivation(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * torch.sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid = torch.sigmoid(input)
        return grad_output * (sigmoid * (1 + input * (1 - sigmoid)))

def swish(x):
    return CustomActivation.apply(x)

class CustomSwishActivation(nn.Module):
    def forward(self, x):
        return swish(x)

class CustomPooling(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 2) + F.max_pool2d(x, 2)


# Residual блоки

class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels=8):
        super(BottleneckResidualBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels, bottleneck_channels, 1)
        self.conv = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)
        self.expand = nn.Conv2d(bottleneck_channels, in_channels, 1)

    def forward(self, x):
        residual = x
        out = F.relu(self.reduce(x))
        out = F.relu(self.conv(out))
        out = self.expand(out)
        return F.relu(out + residual)

class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, widen_factor=2):
        super(WideResidualBlock, self).__init__()
        out_channels = in_channels * widen_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


#  мини-сеть с одним блоком

class MiniResNet(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.block = block
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)