import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):


    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):


    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):


    def __init__(self, in_ch=1, out_ch=1, features=None):
        super().__init__()
        features = features or [64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2)


        self.enc = nn.ModuleList([
            ConvBlock(in_ch,       features[0]),
            ConvBlock(features[0], features[1]),
            ConvBlock(features[1], features[2]),
            ConvBlock(features[2], features[3]),
        ])


        self.bottleneck = ConvBlock(features[3], features[3] * 2)


        self.dec = nn.ModuleList([
            UpBlock(features[3] * 2, features[3]),
            UpBlock(features[3],     features[2]),
            UpBlock(features[2],     features[1]),
            UpBlock(features[1],     features[0]),
        ])


        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []


        for enc_block in self.enc:
            x = enc_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]


        for i, dec_block in enumerate(self.dec):
            x = dec_block(x, skips[i])

        return self.final_conv(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_ch=1, out_ch=1)
    x     = torch.randn(2, 1, 256, 256)
    out   = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {count_parameters(model):,}")
    assert out.shape == x.shape, "Output shape must match input shape!"
    print("Sanity check passed!")
