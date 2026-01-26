"""
Image Transformation Network (Generator)
Johnson et al. architecture with InstanceNorm
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with instance normalization and reflection padding"""
    def __init__(self, channels):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.pad1(x)
        out = self.relu(self.in1(self.conv1(out)))
        out = self.pad2(out)
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class TransformerNetwork(nn.Module):
    """
    Image Transformation Network for fast neural style transfer
    Input/Output: (B, 3, 256, 256)
    """
    def __init__(self):
        super().__init__()
        
        # Downsampling encoder with reflection padding
        self.pad1 = nn.ReflectionPad2d(4)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=0)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        
        # Residual bottleneck (5 blocks)
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # Upsampling decoder (using Upsample + Conv instead of ConvTranspose2d)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.pad5 = nn.ReflectionPad2d(1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        
        # Output layer
        self.pad_out = nn.ReflectionPad2d(4)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Encoder
        out = self.pad1(x)
        out = self.relu(self.in1(self.conv1(out)))
        
        out = self.pad2(out)
        out = self.relu(self.in2(self.conv2(out)))
        
        out = self.pad3(out)
        out = self.relu(self.in3(self.conv3(out)))
        
        # Residual blocks
        out = self.res_blocks(out)
        
        # Decoder
        out = self.upsample1(out)
        out = self.pad4(out)
        out = self.relu(self.in4(self.conv4(out)))
        
        out = self.upsample2(out)
        out = self.pad5(out)
        out = self.relu(self.in5(self.conv5(out)))
        
        # Output - tanh gives [-1, 1], scale to [0, 1]
        out = self.pad_out(out)
        out = self.conv_out(out)
        out = self.tanh(out)
        out = (out + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        
        # Clamp to ensure [0, 1] range (safety check)
        out = torch.clamp(out, 0.0, 1.0)
        
        return out