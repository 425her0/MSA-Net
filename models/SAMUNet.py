import torch
import torch.nn as nn
from nets.init_weights import init_weights
from nets.modules import ShuffleV2Block, unetUp

class SAMUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=21, feature_scale=1, cbam=True, msff=True):
        super(SAMUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.feature_scale = feature_scale
        self.cbam = cbam
        self.msff = msff
        self.num_repeats = [1, 3, 4, 6, 3]
        self.filters = [64, 64, 128, 256, 512, 1024]
        self.filters = [int(x / self.feature_scale) for x in self.filters]

        # Preprocessing
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, self.filters[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.filters[0]),
            nn.ReLU(inplace=True))

        # Encoder
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.en_block_0, self.en_block_1, self.en_block_2, self.en_block_3, self.en_block_4 = [], [], [], [], []
        self.en_block = ["self.en_block_0", "self.en_block_1", "self.en_block_2", "self.en_block_3", "self.en_block_4"]

        for idx in range(0, len(self.num_repeats)):
            repeat = self.num_repeats[idx]
            if idx == 0:
                input_channel = self.filters[0]
                output_channel = self.filters[0]

            else:
                input_channel = self.filters[idx]
                output_channel = self.filters[idx+1]

            for i in range(repeat):
                eval(self.en_block[idx]).append(ShuffleV2Block(input_channel//2, output_channel,
                                                    mid_channels=output_channel//2, ksize=3, stride=1))
                input_channel = output_channel

        self.en_block_0 = nn.Sequential(*self.en_block_0)
        self.en_block_1 = nn.Sequential(*self.en_block_1)
        self.en_block_2 = nn.Sequential(*self.en_block_2)
        self.en_block_3 = nn.Sequential(*self.en_block_3)
        self.en_block_4 = nn.Sequential(*self.en_block_4)

        self.maxpool_1 = nn.MaxPool2d(16, 16, ceil_mode=True)
        self.maxpool_2 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.maxpool_3 = nn.MaxPool2d(4, 4, ceil_mode=True)

        self.en_fuse_conv = nn.Conv2d(sum(self.filters[1:-1]), self.filters[-2], 1, 1, 0, bias=False)
        self.en_fuse_norm = nn.BatchNorm2d(self.filters[-2])
        self.en_fuse_relu = nn.ReLU(inplace=True)

        # Decoder
        self.de_block_0, self.de_block_1, self.de_block_2, self.de_block_3 = [], [], [], []
        self.de_block = ["self.de_block_0", "self.de_block_1", "self.de_block_2", "self.de_block_3"]

        for idx in range(4):
            input_channel  = self.filters[2:][3-idx] // 2
            output_channel = self.filters[1:5][3-idx]

            for i in range(self.num_repeats[3-idx]):
                eval(self.de_block[3-idx]).append(ShuffleV2Block(input_channel//2, output_channel,
                                                    mid_channels=output_channel//2, ksize=3, stride=1))
                input_channel = output_channel

        self.de_block_0 = nn.Sequential(*self.de_block_0)
        self.de_block_1 = nn.Sequential(*self.de_block_1)
        self.de_block_2 = nn.Sequential(*self.de_block_2)
        self.de_block_3 = nn.Sequential(*self.de_block_3)

        self.up_concat_3 = unetUp(self.filters[4], self.filters[-1], self.filters[4], self.cbam)
        self.up_concat_2 = unetUp(self.filters[3], self.filters[-2], self.filters[3], self.cbam)
        self.up_concat_1 = unetUp(self.filters[2], self.filters[-3], self.filters[2], self.cbam)
        self.up_concat_0 = unetUp(self.filters[1],  self.filters[-4], self.filters[1], self.cbam)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.de_fuse_conv = nn.Conv2d(sum(self.filters[1:5]), self.filters[1], 1, 1, 0, bias=False)
        self.de_fuse_norm = nn.BatchNorm2d(self.filters[1])
        self.de_fuse_relu = nn.ReLU(inplace=True)

        # Output
        self.final = nn.Conv2d(self.filters[1], self.n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        # downsampling
        x = self.conv(x)
        feat_0 = self.en_block_0(x)

        x = self.downsample(feat_0)
        feat_1 = self.en_block_1(x)

        x = self.downsample(feat_1)
        feat_2 = self.en_block_2(x)

        x = self.downsample(feat_2)
        feat_3 = self.en_block_3(x)

        x = self.downsample(feat_3)
        if self.msff:
            en_fuse = self.en_fuse_relu(self.en_fuse_norm(self.en_fuse_conv(
                torch.cat((self.maxpool_1(feat_0), self.maxpool_2(feat_1), self.maxpool_3(feat_2), x), dim=1))))
            feat_4 = self.en_block_4(en_fuse)
        else:
            feat_4 = self.en_block_4(x)


        # upsampling
        up_3_ = self.up_concat_3(feat_3, feat_4)
        up_3 = self.de_block_3(up_3_)

        up_2_ = self.up_concat_2(feat_2, up_3)
        up_2 = self.de_block_2(up_2_)

        up_1_ = self.up_concat_1(feat_1, up_2)
        up_1 = self.de_block_1(up_1_)

        up_0_ = self.up_concat_0(feat_0, up_1)
        if self.msff:
            de_fuse = self.de_fuse_relu(self.de_fuse_norm(self.de_fuse_conv(
                torch.cat((up_0_, self.upsample_2(up_1), self.upsample_4(up_2), self.upsample_8(up_3)), dim=1))))
            up_0 = self.de_block_0(de_fuse)
        else:
            up_0 = self.de_block_0(up_0_)

        # output
        final = self.final(up_0)
        return final






