import torch
from torch import nn
import torch.nn.functional as F

# 搭建网络
AlexNet = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(9216, 4096), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.Dropout(p=0.5),
    nn.Linear(4096, 1000), nn.Dropout(p=0.5),
    nn.Linear(1000, 120)
)

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.ReLU = nn.ReLU()
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.s3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(9216, 4096)
        self.f7 = nn.Linear(4096, 4096)
        self.f8 = nn.Linear(4096, 1000)
        self.f9 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s3(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)

        return x

# 测试网络
if __name__ == '__main__':
    input = torch.randn(size=(1, 3, 223, 223), dtype=torch.float32)
    for layer in AlexNet:
        input = layer(input)
        print(layer.__class__.__name__, 'output shape: \t', input.shape)
