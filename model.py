import torch
from torch import nn
from torch.nn import functional as funct


class DualConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),  # or delete
            nn.LeakyReLU(),  # 'nn.ReLU()' or 'nn.LeakyReLU()'
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),  # or delete
            nn.LeakyReLU()  # 'nn.ReLU()' or 'nn.LeakyReLU()'
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()  # 'nn.ReLU()' or 'nn.LeakyReLU()'
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(channel, channel//2, kernel_size=1, stride=1)

    def forward(self, x, feature_map):
        out = self.conv(funct.interpolate(x, scale_factor=2, mode='nearest'))
        return torch.cat((out, feature_map), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = DualConv(1, 64)
        self.d1 = DownSample(64)
        self.c2 = DualConv(64, 128)
        self.d2 = DownSample(128)
        self.c3 = DualConv(128, 256)
        self.d3 = DownSample(256)
        self.c4 = DualConv(256, 512)
        self.d4 = DownSample(512)
        self.c5 = DualConv(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = DualConv(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = DualConv(512, 256)
        self.u3 = UpSample(256)
        self.c8 = DualConv(256, 128)
        self.u4 = UpSample(128)
        self.c9 = DualConv(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))
        r5 = self.c5(self.d4(r4))
        o1 = self.c6(self.u1(r5, r4))
        o2 = self.c7(self.u2(o1, r3))
        o3 = self.c8(self.u3(o2, r2))
        o4 = self.c9(self.u4(o3, r1))
        return self.out(o4)
