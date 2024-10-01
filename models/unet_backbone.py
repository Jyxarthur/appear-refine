import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class UNet_encoder_with_masks_for_corrector(nn.Module):
    """ input dimension: b c h w (batch_size channel height width)"""
    """ output dimension: b c' h' w' """
    def __init__(self, in_channels, init_features):
        super(UNet_encoder_with_masks_for_corrector, self).__init__()
        features = init_features
        self.encoder1_rgb = ConvBlock(in_channels, features, name="enc1_r")
        self.encoder1_mask = ConvBlock(3, features, name="enc1_m")
        self.encoder1 = ConvBlock(features * 2, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(features * 4, features * 8, name="bottleneck")

    def forward(self, rgbs, masks):
        enc1_rgbs = self.encoder1_rgb(rgbs) # (b t) c h w
        enc1_masks = self.encoder1_mask(masks) # (b t) c h w
        x = torch.cat((enc1_rgbs, enc1_masks), 1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        return enc1, enc2, enc3, bottleneck



class UNet_encoder_with_masks_for_selector(nn.Module):
    """ input dimension: b c h w (batch_size channel height width)"""
    """ output dimension: b c' h' w' """
    def __init__(self, in_channels=3, init_features=64):
        super(UNet_encoder_with_masks_for_selector, self).__init__()
        features = init_features
        self.encoder1_rgb = ConvBlock(in_channels, features, name="enc1_r")
        self.encoder1_mask = ConvBlock(1, features, name="enc1_m")
        self.encoder1 = ConvBlock(features * 2, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(features * 8, features * 16, name="bottleneck")

    def forward(self, rgbs, masks):
        enc1_rgbs = self.encoder1_rgb(rgbs)
        enc1_masks = self.encoder1_mask(masks) # (b t q) c h w
        # Combine RGB and masks at the first level
        x = torch.cat((enc1_rgbs, enc1_masks), 1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        return enc1, enc2, enc3, enc4, bottleneck


class UNet_decoder_with_mask(nn.Module):
    """ input dimension: b c' h' w' (batch_size channel height width)"""
    """ output dimension: b c h w"""
    def __init__(self, out_channels, init_features):
        super(UNet_decoder_with_mask, self).__init__()
        features = init_features
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ConvBlock((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ConvBlock((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = ConvBlock(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        
    def forward(self, enc1_rgbmask, enc2_rgbmask, enc3_rgbmask, bottleneck):
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3_rgbmask), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2_rgbmask), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1_rgbmask), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)



class UNet_decoder(nn.Module):
    """ input dimension: b c' h' w' (batch_size channel height width)"""
    """ output dimension: b c h w"""
    def __init__(self, out_channels=1, init_features=64):
        super(UNet_decoder, self).__init__()
        features = init_features
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = ConvBlock((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ConvBlock((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ConvBlock((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = ConvBlock(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
    def forward(self, enc1, enc2, enc3, enc4, bottleneck):
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)

        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
  
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)

        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    

class ConvBlock(nn.Module):
    """ Double convolution block used in U-Net"""
    def __init__(self, in_channels, features, name):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    #(name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    #(name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(inplace=True)),
                ]
            )
        )
        
    def forward(self, x):
        return self.conv_block(x)
        
