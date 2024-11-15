import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class DirectCurrent(nn.Module):
    def __init__(self, dim):
        super(DirectCurrent, self).__init__()

        self.weight = nn.Parameter(torch.ones(dim, 1, 1))
        self.fuse = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        res = x.clone()
        dc = F.adaptive_avg_pool2d(x, (1,1))
        remain = x - dc
        dc = dc * self.weight
        out = self.fuse(remain + dc)
        return res + out


class PoolDC(nn.Module):
    def __init__(self, k, k_out):
        super(PoolDC, self).__init__()
        self.pools_sizes = [8,4,2]
        pools, dcs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            dcs.append(DirectCurrent(k))
        self.pools = nn.ModuleList(pools)

        self.dcs = nn.ModuleList(dcs)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x

        y = self.dcs[0](self.pools[0](x))
        resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        y_up1 = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)

        y = self.dcs[1](self.pools[1](x)+y_up1)
        resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        y_up2 = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        
        y_up1 = F.interpolate(y_up1, scale_factor=2, mode='bilinear', align_corners=True)

        y = self.dcs[2](self.pools[2](x)+y_up2+y_up1)
        resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dc=False):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            PoolDC(in_channel, out_channel) if dc else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x