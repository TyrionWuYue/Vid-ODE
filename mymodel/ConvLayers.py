import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Sequential):
    def __init__(self, channels, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Conv2d(channels[i-1], channels[i], 1, bias=bias, groups=4))
            m.append(nn.ReLU())
            if drop > 0:
                m.append(nn.Dropout2d(drop))
        super(BasicConv, self).__init__(*m)
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            nn.ReLU(),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Encoder(nn.Module):
    
    def __init__(self, input_dim=3, ch=64, n_downs=2):
        super(Encoder, self).__init__()
        
        model = []
        model += [nn.Conv2d(input_dim, ch, 3, 1, 1)]
        model += [nn.BatchNorm2d(ch)]
        model += [nn.ReLU()]
        
        for _ in range(n_downs):
            model += [nn.Conv2d(ch, ch * 2, 4, 2, 1)]
            model += [nn.BatchNorm2d(ch * 2)]
            model += [nn.ReLU()]
            ch *= 2
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        out = self.model(x)
        return out


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim=3, n_ups=2):
        super(Decoder, self).__init__()
        model = []
        ch = input_dim
        for i in range(n_ups):
            model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            model += [nn.Conv2d(ch, ch//2, 3, 1, 1)]
            model += [nn.BatchNorm2d(ch//2)]
            model += [nn.ReLU()]
            ch = ch // 2
        model += [nn.Conv2d(ch, output_dim, 3, 1, 1)]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        out = self.model(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)