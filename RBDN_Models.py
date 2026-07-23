import torch
from torch import nn
import torch.nn.functional as F

# this module is based on "Generalized Deep Image to Image Regression"
# Created by chatgpt
class ConvReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

class DeconvBranch(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class RBDNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvReLU(channels, channels)
        self.conv2 = ConvReLU(channels, channels)
        self.deconv_branch = DeconvBranch(channels, channels, channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.deconv_branch(x)
        return x + F.interpolate(identity, scale_factor=4, mode='bilinear', align_corners=False)

class RBDN(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_blocks=3):
        super().__init__()
        self.entry = ConvReLU(in_channels, num_features)
        self.blocks = nn.Sequential(*[RBDNBlock(num_features) for _ in range(num_blocks)])
        self.reconstruction = nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.reconstruction(x)
        return x

# --- Client Test Code ---
def test_rbdn():
    model = RBDN()
    model.eval()

    input_tensor = torch.randn(1, 3, 32, 32)  # Simulate a low-res image
    with torch.no_grad():
        output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)


class BlockForward(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, include_pool=True, use_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Tanh()
        self.include_pool = include_pool
        if include_pool:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        else:
            self.pool = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.include_pool:
            x, indices = self.pool(x)
        else:
            indices = None
        return x, indices

class BlockBackward(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, use_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Tanh()
        self.unpool = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.act2 = nn.ReLU(inplace=True)
        else:
            self.act2 = nn.Tanh()

    def forward(self, x, indices, output_size=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.unpool(x, indices, output_size=output_size)
        x = self.deconv(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class FinalConvs(nn.Module):
    def __init__(self, n_channels, kernel_size=3, stride=1,
                 padding=1):
        super().__init__()
        self.layers = nn.ModuleList(
                        [BlockForward(n_channels, n_channels, kernel_size, stride, padding, False)
                        for _ in range(9)]
                                    )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
        return x

class Output(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True):
        super().__init__()
        self.unpool = torch.nn.MaxUnpool2d(2, stride=2)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x, indices, output_size=None):
        x = self.unpool(x, indices, output_size=output_size)
        x = self.deconv(x)
        return x

class myRBDN0(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, out_channels=3):
        super().__init__()
        self.B0 = BlockForward(in_channels, mid_channels, 9, padding=4, include_pool=True)
        self.BF1 = BlockForward(mid_channels, mid_channels, 3, padding=1, include_pool=True)
        self.BF2 = BlockForward(mid_channels, mid_channels, 3, padding=1, include_pool=True)
        self.BF3 = BlockForward(mid_channels, mid_channels, 3, padding=1, include_pool=True)

        self.BB3 = BlockBackward(mid_channels, mid_channels, 3, padding=1)
        self.BB2 = BlockBackward(2 * mid_channels, mid_channels, 3, padding=1)
        self.BB1 = BlockBackward(2 * mid_channels, mid_channels, 3, padding=1)

        self.final_convs = FinalConvs(mid_channels, 3)

        self.output = Output(mid_channels, out_channels, 3, padding=1)

    def forward(self, x):
        y, ind0 = self.B0(x)
        y1, ind1 = self.BF1(y)
        y2, ind2 = self.BF2(y1)
        y3, ind3 = self.BF3(y2)

        z3 = self.BB3(y3, ind3)
        yc2 = torch.cat((z3, y2), dim=1)
        z2 = self.BB2(yc2, ind2)
        yc1 = torch.cat((z2, y1), dim=1)
        z1 = self.BB1(yc1, ind1)
        z0 = self.final_convs(z1)
        z0 = self.output(z0, ind0)
        return z0

class myRBDN(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, out_channels=3, n_branches=3, use_relu=True, output_bias=True):
        super().__init__()
        self.use_relu = use_relu
        self.n_branches = n_branches
        self.BF0 = BlockForward(in_channels, mid_channels, 9, padding=4, include_pool=True, use_relu=use_relu)
        self.BFs = nn.ModuleList(
                    [BlockForward(mid_channels, mid_channels, 3, padding=1, include_pool=True, use_relu=use_relu)
                    for _ in range(n_branches)]
                                )

        self.BB0 = BlockBackward(mid_channels, mid_channels, 3, padding=1, use_relu=use_relu)
        self.BBs = nn.ModuleList(
                    [BlockBackward(2 * mid_channels, mid_channels, 3, padding=1, use_relu=use_relu)
                    for _ in range(n_branches-1)]
                                )

        self.final_convs = FinalConvs(mid_channels, 3)

        self.output = Output(mid_channels, out_channels, 3, padding=1, bias=output_bias)

    def forward(self, x):
        inds = []
        shapes = []
        ys = []
        y, ind0 = self.BF0(x)
        shape0 = x.shape[2:]
        for i, bf in enumerate(self.BFs):
            shapes.append(y.shape[2:])
            y, ind = bf(y)
            ys.append(y)
            inds.append(ind)

        z = self.BB0(ys[-1], inds[-1], shapes[-1])
        last_ix = self.n_branches - 2
        for i, bb in enumerate(self.BBs):
            yc = torch.cat((z, ys[last_ix]), dim=1)
            z = bb(yc, inds[last_ix], shapes[last_ix])
            last_ix -= 1

        z = self.final_convs(z)
        z = self.output(z, ind0, shape0)
        if not self.use_relu:
            z = nn.Tanh()(z)
        return z

# prompt: write a class called myRBDN2 similar to the myRBDN class currently defined to use magnify the input with interpolation and follow it with a convolution layer instead of each unpool+deconv pair

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, padding=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class BlockBackward2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample_conv = UpsampleConv(out_channels, out_channels, scale_factor=scale_factor, kernel_size=kernel_size, padding=padding)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample_conv(x)
        return x

class Output2(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, padding=1):
        super().__init__()
        self.upsample_conv = UpsampleConv(in_channels, out_channels, scale_factor=scale_factor, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.upsample_conv(x)
        return x


class myRBDN2(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, out_channels=3, n_branches=3):
        super().__init__()
        self.n_branches = n_branches
        self.BF0 = BlockForward(in_channels, mid_channels, 9, padding=4, include_pool=True)
        self.BFs = nn.ModuleList(
                    [BlockForward(mid_channels, mid_channels, 3, padding=1, include_pool=True)
                    for _ in range(n_branches)]
                                )

        self.BB0 = BlockBackward2(mid_channels, mid_channels, scale_factor=2, kernel_size=3, padding=1)
        # Note: BlockBackward2 does not need indices
        self.BBs = nn.ModuleList(
                    [BlockBackward2(mid_channels + mid_channels, mid_channels, scale_factor=2, kernel_size=3, padding=1)
                    for _ in range(n_branches-1)]
                                )

        self.final_convs = FinalConvs(mid_channels, 3)

        self.output = Output2(mid_channels, out_channels, scale_factor=2, kernel_size=3, padding=1)

    def forward(self, x):
        ys = []
        y, _ = self.BF0(x) # We don't need indices anymore
        for i, bf in enumerate(self.BFs):
            y, _ = bf(y) # We don't need indices anymore
            ys.append(y)

        z = self.BB0(ys[-1])
        last_ix = self.n_branches - 2
        for i, bb in enumerate(self.BBs):
            yc = torch.cat((z, ys[last_ix]), dim=1)
            z = bb(yc)
            last_ix -= 1

        z = self.final_convs(z)
        z = self.output(z)
        return z

# --- Client Test Code ---
def test_myrbdn2():
    model = myRBDN2(n_branches=3)
    model.eval()

    input_tensor = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

# prompt: initialize all weights of conv layers weights to Gaussian (mean=0, std=0.001) and biases to zero. also batch norm scales to 1 and shifts to 0.001

def initialize_rbdn(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.001)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0.001)

