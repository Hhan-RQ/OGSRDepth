
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from ProDepth.layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales
        self.num_scales = len(scales)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_scales, -1, -1):
            self.convs[("upconv", i, 4)]=MA_EdgeContext(self.num_ch_enc[i])

        for i in range(self.num_scales, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == self.num_scales else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs[("upconv", i, 2)] = VSSBlock(num_ch_out)
            self.convs[("upconv", i, 3)] = MFAR2(num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, is_multi=False):
        self.outputs = {}

        for i in range(self.num_scales, -1, -1):
            input_features[i] = self.convs[("upconv", i, 4)](input_features[i]) # ABCM

        x = input_features[-1]

        for i in range(self.num_scales, -1, -1):
            x = self.convs[("upconv", i, 0)](x) # (B,C,H,W)
            x1 = self.convs[("upconv", i, 3)](x) # MSCA
            x = x.permute(0, 2, 3, 1) # (B,H,W,C)
            x2 = self.convs[("upconv", i, 2)](x) # SGMB
            x2 = x2.permute(0, 3, 1, 2)
            x = x1 + x2
            if i == 0 and is_multi == True: # 不满足is_multi == True
                x = [upsample2(x)]
            else:
                x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
