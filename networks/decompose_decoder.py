import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import *

class DecomposeDecoder(nn.Module):
    """
    Non-Lambertian decomposition decoder that outputs three components:
    - Albedo (A): material reflectance properties
    - Shading (S): illumination effects  
    - Mask (M): occlusion effects
    Following the model: I = A × S + M × I
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, use_skips=True):
        super(DecomposeDecoder, self).__init__()
        
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 64, 128, 256])

        self.convs = OrderedDict()
        
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_A", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_A", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.convs[("decompose_A_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.convs["S_M_branch"] = ConvBlock(3, self.num_ch_dec[0])
        self.convs[("decompose_S_conv", 0)] = Conv3x3(2 * self.num_ch_dec[0], 1)

        self.convs[("M_branch", 0)] = ConvBlock(3, self.num_ch_dec[0])
        self.convs[("decompose_M_conv", 0)] = Conv3x3(3 * self.num_ch_dec[0], 1)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features, images):
        self.outputs = {}
        # Albedo decoder (A)
        x_A = input_features[-1]
        for i in range(4, -1, -1):
            x_A = self.convs[("upconv_A", i, 0)](x_A)
            x_A = [upsample(x_A)]
            if self.use_skips and i > 0:
                x_A += [input_features[i - 1]]
            x_A = torch.cat(x_A, 1)
            x_A = self.convs[("upconv_A", i, 1)](x_A)
           
        x_S = [self.convs["S_M_branch"](images)]
        x_S += [x_A]
        x_S = torch.cat(x_S, 1)
        
        x_M = self.convs[("M_branch", 0)](images)
        x_M = torch.cat([x_M, x_S], 1)
    
        self.outputs["A"] = self.sigmoid(self.convs[("decompose_A_conv", 0)](x_A))
        self.outputs["S"] = self.sigmoid(self.convs[("decompose_S_conv", 0)](x_S))
        self.outputs["M"] = self.sigmoid(self.convs[("decompose_M_conv", 0)](x_M))
        return self.outputs