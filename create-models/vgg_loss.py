"""
VGG-based perceptual loss network
Extracts features for content and style losses
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from collections import namedtuple


class VGGLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    Content: relu2_2
    Style: relu1_2, relu2_2, relu3_3, relu4_3
    """

    def __init__(self):
        super().__init__()

        # Load pretrained VGG19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        # Layer mapping for ReLU outputs
        # VGG features indices: conv1_1(0), relu1_1(1), conv1_2(2), relu1_2(3), pool1(4)
        # conv2_1(5), relu2_1(6), conv2_2(7), relu2_2(8), pool2(9)
        # conv3_1(10), relu3_1(11), conv3_2(12), relu3_2(13), conv3_3(14), relu3_3(15), conv3_4(16), relu3_4(17), pool3(18)
        # conv4_1(19), relu4_1(20), conv4_2(21), relu4_2(22), conv4_3(23), relu4_3(24), ...

        self.slice1 = nn.Sequential()  # relu1_2
        self.slice2 = nn.Sequential()  # relu2_2
        self.slice3 = nn.Sequential()  # relu3_3
        self.slice4 = nn.Sequential()  # relu4_3

        # relu1_2 is at index 3
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])

        # relu2_2 is at index 8
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])

        # relu3_3 is at index 15
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])

        # relu4_3 is at index 24
        for x in range(16, 25):
            self.slice4.add_module(str(x), vgg[x])

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        """Normalize image with ImageNet mean/std"""
        return (x - self.mean) / self.std

    def forward(self, x):
        """
        Extract features from all relevant layers
        Returns: namedtuple with relu1_2, relu2_2, relu3_3, relu4_3
        """
        x = self.normalize(x)

        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)

        VGGOutputs = namedtuple(
            "VGGOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        )
        return VGGOutputs(h1, h2, h3, h4)


def gram_matrix(feat):
    """
    Compute Gram matrix for style representation
    Args:
        feat: (B, C, H, W)
    Returns:
        gram: (B, C, C)
    """
    B, C, H, W = feat.size()
    feat = feat.view(B, C, H * W)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / (C * H * W)  # Normalize by number of elements


class StyleTransferLoss(nn.Module):
    """
    Combined content, style, and TV losses
    """

    def __init__(self, content_weight=1.0, style_weight=5e10, tv_weight=1e-5):
        super().__init__()
        self.vgg = VGGLoss()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.mse = nn.MSELoss()

        # Cache for style gram matrices
        self.style_grams = None

    def set_style_image(self, style_img):
        """
        Precompute and cache style Gram matrices
        Args:
            style_img: (1, 3, H, W) or (B, 3, H, W)
        """
        with torch.no_grad():
            style_feats = self.vgg(style_img)
            self.style_grams = [
                gram_matrix(style_feats.relu1_2),
                gram_matrix(style_feats.relu2_2),
                gram_matrix(style_feats.relu3_3),
                gram_matrix(style_feats.relu4_3),
            ]

    def content_loss(self, generated_feats, content_feats):
        """Content loss at relu2_2"""
        return self.mse(generated_feats.relu2_2, content_feats.relu2_2)

    def style_loss(self, generated_feats):
        """
        Style loss across multiple layers
        Compares to cached style grams
        """
        if self.style_grams is None:
            raise RuntimeError(
                "Must call set_style_image() before computing style loss"
            )

        generated_grams = [
            gram_matrix(generated_feats.relu1_2),
            gram_matrix(generated_feats.relu2_2),
            gram_matrix(generated_feats.relu3_3),
            gram_matrix(generated_feats.relu4_3),
        ]

        loss = 0
        for gen_gram, style_gram in zip(generated_grams, self.style_grams):
            # Expand style gram to match batch size if needed
            if style_gram.size(0) == 1 and gen_gram.size(0) > 1:
                style_gram = style_gram.expand_as(gen_gram)
            loss += self.mse(gen_gram, style_gram)

        return loss

    def tv_loss(self, img):
        """
        Total variation loss for smoothness
        Args:
            img: (B, C, H, W)
        """
        batch_size, _, h, w = img.size()

        # Horizontal variation
        h_tv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()

        # Vertical variation
        w_tv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()

        return (h_tv + w_tv) / (batch_size * h * w)

    def forward(self, generated, content):
        """
        Compute total loss
        Args:
            generated: (B, 3, H, W) - output from transformer network
            content: (B, 3, H, W) - original content images
        Returns:
            total_loss, content_loss, style_loss, tv_loss
        """
        # Extract features
        gen_feats = self.vgg(generated)
        content_feats = self.vgg(content)

        # Compute individual losses
        c_loss = self.content_loss(gen_feats, content_feats)
        s_loss = self.style_loss(gen_feats)
        tv_loss = self.tv_loss(generated)

        # Weighted sum
        total = (
            self.content_weight * c_loss
            + self.style_weight * s_loss
            + self.tv_weight * tv_loss
        )

        return total, c_loss, s_loss, tv_loss
