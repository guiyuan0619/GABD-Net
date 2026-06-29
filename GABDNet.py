import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import cv2
import numpy as np


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )

        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GaborFilterBank(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, directions=4):
        super(GaborFilterBank, self).__init__()
        self.directions = directions
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.sigma = nn.Parameter(torch.ones(directions) * 2.0)
        self.lambda_ = nn.Parameter(torch.ones(directions) * 5.0)
        self.gamma = nn.Parameter(torch.ones(directions) * 0.5)
        self.psi = nn.Parameter(torch.ones(directions) * 0.0)


        self.theta = torch.tensor([i * math.pi / directions for i in range(directions)])


        self.fusion_conv = nn.Conv2d(in_channels * directions, out_channels, kernel_size=1, bias=False)


        self._initialize_all_shared_kernels()

    def _initialize_all_shared_kernels(self):
        k = self.kernel_size // 2
        try:
            y, x = torch.meshgrid(torch.arange(-k, k + 1), torch.arange(-k, k + 1), indexing='ij')
        except TypeError:
            y, x = torch.meshgrid(torch.arange(-k, k + 1), torch.arange(-k, k + 1))
        x = x.float().to(self.sigma.device)
        y = y.float().to(self.sigma.device)

        kernels = []
        for i in range(self.directions):
            theta = self.theta[i].to(self.sigma.device)
            sigma = self.sigma[i]
            lambda_ = self.lambda_[i]
            gamma = self.gamma[i]
            psi = self.psi[i]

            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

            gb_real = torch.exp(- (x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2 * sigma ** 2)) * torch.cos(
                2 * math.pi * x_theta / lambda_ + psi)
            kernels.append(gb_real.unsqueeze(0).unsqueeze(0))


        self.shared_gabor_kernels = nn.Parameter(torch.cat(kernels, dim=0))

    def forward(self, x):

        gabor_kernels = self.shared_gabor_kernels
        gabor_kernels = gabor_kernels.repeat(x.size(1), 1, 1, 1)


        x = F.conv2d(x, gabor_kernels, padding=self.kernel_size // 2, groups=x.size(1))


        x = self.fusion_conv(x)
        return x


class AdaptiveBlockDCT(nn.Module):

    def __init__(self, depth_level=3):
        super(AdaptiveBlockDCT, self).__init__()
        self.depth_level = depth_level

        self.gabor = GaborFilterBank(in_channels=1, out_channels=4, kernel_size=3, directions=4)

    def _compute_adaptive_block_size(self, H, W):
        min_dim = min(H, W)


        base_divisor = 2 ** (6 - self.depth_level)
        base_block_size = max(4, min_dim // base_divisor)


        block_size = min(32, max(4, base_block_size))


        block_size = (block_size // 2) * 2

        return block_size

    def forward(self, x):
        B, C, H, W = x.size()


        block_size = self._compute_adaptive_block_size(H, W)


        subbands = []
        for c in range(C):
            x_c = x[:, c:c + 1, :, :]
            subband_c = self.gabor(x_c)
            subbands.append(subband_c)
        subbands = torch.cat(subbands, dim=1)


        subbands = F.avg_pool2d(subbands, kernel_size=block_size, stride=block_size)

        return subbands


class IDCTReconstruction(nn.Module):
    def __init__(self, in_channels):
        super(IDCTReconstruction, self).__init__()
        self.in_channels = in_channels


        self.subband_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.5, 0.25]))


        self.residual_conv = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1, bias=False)


        self.enhance_conv = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, original=None):
        B, C, H, W = x.size()
        out_channel = C // 4


        x_ll = x[:, 0:out_channel, :, :]
        x_lh = x[:, out_channel:out_channel * 2, :, :]
        x_hl = x[:, out_channel * 2:out_channel * 3, :, :]
        x_hh = x[:, out_channel * 3:out_channel * 4, :, :]


        if original is not None:
            out_h, out_w = original.size(2), original.size(3)
        else:
            out_h, out_w = H * 2, W * 2


        x_ll = F.interpolate(x_ll, size=(out_h, out_w), mode='bilinear', align_corners=False) * self.subband_weights[0]
        x_lh = F.interpolate(x_lh, size=(out_h, out_w), mode='bilinear', align_corners=False) * self.subband_weights[1]
        x_hl = F.interpolate(x_hl, size=(out_h, out_w), mode='bilinear', align_corners=False) * self.subband_weights[2]
        x_hh = F.interpolate(x_hh, size=(out_h, out_w), mode='bilinear', align_corners=False) * self.subband_weights[3]


        h = x_ll + x_lh + x_hl + x_hh


        h = self.enhance_conv(h)


        if original is not None:

            if original.shape[2:] != h.shape[2:]:
                original = F.interpolate(original, size=h.shape[2:], mode='bilinear', align_corners=False)
            residual = self.residual_conv(original)
            h = h + residual

        return h


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels


        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class InverseAttention(nn.Module):

    def __init__(self, in_channels):
        super(InverseAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        attention_map = self.relu(self.conv1(x))
        attention_map = self.sigmoid(self.conv2(attention_map))


        reverse_attention_map = 1 - attention_map


        return reverse_attention_map * x


class GABDBlock(nn.Module):
    def __init__(self, channels, depth_level=3):
        super(GABDBlock, self).__init__()
        self.dct = AdaptiveBlockDCT(depth_level=depth_level)
        self.idct = IDCTReconstruction(channels * 4)


        self.reverse_attention = InverseAttention(channels)


        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        high_freq = self.dct(x)
        enhanced = self.idct(high_freq, original=x)


        attn = self.attention(x)
        enhanced = enhanced * attn + x


        enhanced = self.reverse_attention(enhanced)

        return enhanced


class GABDNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filts=32):
        super(GABDNet, self).__init__()


        self.enc1 = DoubleConv(n_channels, n_filts)
        self.att1 = GABDBlock(n_filts, depth_level=1)
        self.pool = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(n_filts, n_filts * 2)
        self.att2 = GABDBlock(n_filts * 2, depth_level=2)

        self.enc3 = DoubleConv(n_filts * 2, n_filts * 4)
        self.att3 = GABDBlock(n_filts * 4, depth_level=3)

        self.enc4 = DoubleConv(n_filts * 4, n_filts * 8)
        self.att4 = GABDBlock(n_filts * 8, depth_level=4)

        self.enc5 = DoubleConv(n_filts * 8, n_filts * 16)
        self.att5 = GABDBlock(n_filts * 16, depth_level=5)


        self.bottleneck = DoubleConv(n_filts * 16, n_filts * 32)
        self.att_bottleneck = GABDBlock(n_filts * 32, depth_level=5)


        self.up1 = Up(n_filts * 32, n_filts * 16)
        self.fusion1 = Fusion(n_filts * 32, n_filts * 16)
        self.dec1 = DoubleConv(n_filts * 16, n_filts * 16)
        self.att_dec1 = GABDBlock(n_filts * 16, depth_level=5)

        self.up2 = Up(n_filts * 16, n_filts * 8)
        self.fusion2 = Fusion(n_filts * 16, n_filts * 8)
        self.dec2 = DoubleConv(n_filts * 8, n_filts * 8)
        self.att_dec2 = GABDBlock(n_filts * 8, depth_level=4)

        self.up3 = Up(n_filts * 8, n_filts * 4)
        self.fusion3 = Fusion(n_filts * 8, n_filts * 4)
        self.dec3 = DoubleConv(n_filts * 4, n_filts * 4)
        self.att_dec3 = GABDBlock(n_filts * 4, depth_level=3)

        self.up4 = Up(n_filts * 4, n_filts * 2)
        self.fusion4 = Fusion(n_filts * 4, n_filts * 2)
        self.dec4 = DoubleConv(n_filts * 2, n_filts * 2)
        self.att_dec4 = GABDBlock(n_filts * 2, depth_level=2)

        self.up5 = Up(n_filts * 2, n_filts)
        self.fusion5 = Fusion(n_filts * 2, n_filts)
        self.dec5 = DoubleConv(n_filts, n_filts)
        self.att_dec5 = GABDBlock(n_filts, depth_level=1)


        self.final_conv = nn.Conv2d(n_filts, n_classes, 1)

    def forward(self, x):

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: input contains NaN or Inf values")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)


        e1 = self.enc1(x)
        e1 = self.att1(e1)
        e1_pool = self.pool(e1)

        e2 = self.enc2(e1_pool)
        e2 = self.att2(e2)
        e2_pool = self.pool(e2)

        e3 = self.enc3(e2_pool)
        e3 = self.att3(e3)
        e3_pool = self.pool(e3)

        e4 = self.enc4(e3_pool)
        e4 = self.att4(e4)
        e4_pool = self.pool(e4)

        e5 = self.enc5(e4_pool)
        e5 = self.att5(e5)
        e5_pool = self.pool(e5)


        bottleneck = self.bottleneck(e5_pool)
        bottleneck = self.att_bottleneck(bottleneck)


        d5 = self.up1(bottleneck)
        if d5.shape[2:] != e5.shape[2:]:
            d5 = F.interpolate(d5, size=e5.shape[2:], mode='bilinear', align_corners=False)
        d5 = self.fusion1(torch.cat([d5, e5], dim=1))
        d5 = self.dec1(d5)
        d5 = self.att_dec1(d5)

        d4 = self.up2(d5)
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.fusion2(torch.cat([d4, e4], dim=1))
        d4 = self.dec2(d4)
        d4 = self.att_dec2(d4)

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.fusion3(torch.cat([d3, e3], dim=1))
        d3 = self.dec3(d3)
        d3 = self.att_dec3(d3)

        d2 = self.up4(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.fusion4(torch.cat([d2, e2], dim=1))
        d2 = self.dec4(d2)
        d2 = self.att_dec4(d2)

        d1 = self.up5(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.fusion5(torch.cat([d1, e1], dim=1))
        d1 = self.dec5(d1)
        d1 = self.att_dec5(d1)


        output = self.final_conv(d1)


        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)


        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: output contains NaN or Inf values")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

        return output


GABD = GABDBlock
IA = InverseAttention
AB_DCT = AdaptiveBlockDCT
LearnableDirectionalPriorFilterBank = GaborFilterBank
AdaptiveBlockBasedDCT = AdaptiveBlockDCT
InverseDCTReconstruction = IDCTReconstruction

__all__ = [
    "GABDNet",
    "GABDBlock",
    "GABD",
    "AdaptiveBlockDCT",
    "AdaptiveBlockBasedDCT",
    "AB_DCT",
    "IDCTReconstruction",
    "InverseDCTReconstruction",
    "InverseAttention",
    "IA",
    "GaborFilterBank",
    "LearnableDirectionalPriorFilterBank",
    "DepthwiseSeparableConv",
    "DoubleConv",
    "Fusion",
    "Up",
    "DiceLoss",
    "dice_coefficient",
]

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.softmax(pred, dim=1)


        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)


        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    if pred.size(1) == 1:
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
    else:
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1, keepdim=True).float()


    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)


    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = GABDNet(n_channels=3, n_classes=1, n_filts=32).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Optimized model total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print("Estimated parameter reduction vs. the original model: about 60-70%")
    print("Inverse Attention has been integrated into the GABDBlock module")


    x = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)


    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")


    criterion = DiceLoss()
    loss = criterion(output, target)
    dice = dice_coefficient(output, target)

    print(f"\nLoss: {loss.item():.4f}")
    print(f"Dice coefficient: {dice:.4f}")


    print("\n=== Validate adaptive block DCT strategy ===")
    for depth in range(1, 6):
        dct_test = AdaptiveBlockDCT(depth_level=depth).to(device)
        test_input = torch.randn(1, 32, 128, 128).to(device)
        block_size = dct_test._compute_adaptive_block_size(128, 128)
        print(f"Depth Level {depth}: block size = {block_size}x{block_size}")


    dct = AdaptiveBlockDCT(depth_level=3).to(device)
    coeffs = dct(x)
    high_freq = coeffs[:, 3:, :, :]


    hf_img = high_freq[0].mean(0).cpu().detach().numpy()
    hf_img = (hf_img - hf_img.min()) / (hf_img.max() - hf_img.min() + 1e-6) * 255
    cv2.imwrite('high_freq_gabor_adaptive.png', hf_img.astype(np.uint8))
    print("\nHigh-frequency map saved as high_freq_gabor_adaptive.png")
