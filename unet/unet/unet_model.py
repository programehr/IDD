""" Full assembly of the parts to form the complete network """
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNetGeneral(nn.Module):
    # the following line creates UNet as a special case of UNetGeneral
    # net = UNetGeneral(3, 10, [64, 128, 256, 512, 1024])
    # fixme problem with even number of mid_channels
    def __init__(self, in_channels, n_classes, use_sigmoid=True, mid_channels=None, bilinear=False):
        super(UNetGeneral, self).__init__()
        if mid_channels is None:
            mid_channels = [64, 128, 256, 512, 1024]
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_sigmoid = use_sigmoid

        self.inc = (DoubleConv(in_channels, mid_channels[0]))
        downs = []
        for i, mid_channel in enumerate(mid_channels[:-2]):
            down = Down(mid_channel, mid_channels[i+1])
            downs.append(down)
        self.downs = nn.ModuleList(downs)
        factor = 2 if bilinear else 1
        down = Down(mid_channels[-2], mid_channels[-1] // factor)
        self.downs.append(down)

        ups = []
        rev_mid_channels = list(reversed(mid_channels))
        for i, mid_channel in enumerate(rev_mid_channels[:-1]):
            up = Up(mid_channel, rev_mid_channels[i+1] // factor, bilinear)
            ups.append(up)
        self.ups = nn.ModuleList(ups)

        self.outc = (OutConv(rev_mid_channels[-1], n_classes))
        self.sigmoid = torch.nn.Sigmoid()
        
    def contracting_path(self, x):
        """
        Return the outputs of the contracting path as a list.
        """
        xs = []
        x1 = self.inc(x)
        xs.append(x1)

        old_x = x1
        for down in self.downs:
            new_x = down(old_x)
            xs.append(new_x)
            old_x = new_x

        return xs

    def forward(self, x):
        x1 = self.inc(x)
        xs = [x1]
        old_x = x1
        for i, down in enumerate(self.downs):
            new_x = down(old_x)
            xs.append(new_x)
            old_x = new_x

        x = self.ups[0](xs[-1], xs[-2])
        for i, up in enumerate(self.ups[1:]):
            x = up(x, xs[-3 - i])

        logits = self.outc(x)
        if self.use_sigmoid:
            out = self.sigmoid(logits)
        else:
            out = logits
        return out

    def use_checkpointing(self):  # todo fix this func
        pass
        # self.inc = torch.utils.checkpoint(self.inc)
        # self.down1 = torch.utils.checkpoint(self.down1)
        # self.down2 = torch.utils.checkpoint(self.down2)
        # self.down3 = torch.utils.checkpoint(self.down3)
        # self.down4 = torch.utils.checkpoint(self.down4)
        # self.up1 = torch.utils.checkpoint(self.up1)
        # self.up2 = torch.utils.checkpoint(self.up2)
        # self.up3 = torch.utils.checkpoint(self.up3)
        # self.up4 = torch.utils.checkpoint(self.up4)
        # self.outc = torch.utils.checkpoint(self.outc)
