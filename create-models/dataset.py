"""
Data loading for content and style images
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class ContentDataset(Dataset):
    """
    Dataset for content images (e.g., COCO)
    Loads all images from a folder
    """

    def __init__(self, root_dir, image_size=256):
        """
        Args:
            root_dir: Directory containing images
            image_size: Size to resize/crop images to
        """
        self.root_dir = Path(root_dir)

        # Get all image files
        self.image_files = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        for ext in valid_extensions:
            self.image_files.extend(list(self.root_dir.rglob(f"*{ext}")))
            self.image_files.extend(list(self.root_dir.rglob(f"*{ext.upper()}")))

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        print(f"Found {len(self.image_files)} content images")

        # Transform pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random other image instead
            return self.__getitem__((idx + 1) % len(self))


class StyleImageLoader:
    """
    Loader for a single style image
    Caches and duplicates to match batch size
    """

    def __init__(self, style_path, image_size=256):
        """
        Args:
            style_path: Path to style image
            image_size: Size to resize/crop to
        """
        self.style_path = Path(style_path)

        if not self.style_path.exists():
            raise FileNotFoundError(f"Style image not found: {style_path}")

        # Load and transform
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

        image = Image.open(self.style_path).convert("RGB")
        self.style_image = transform(image).unsqueeze(0)  # (1, 3, H, W)

        print(f"Loaded style image from {style_path}")

    def get_batch(self, batch_size, device="cpu"):
        """
        Get style image duplicated to match batch size
        Args:
            batch_size: Number of copies
            device: Device to move tensor to
        Returns:
            style_batch: (batch_size, 3, H, W)
        """
        style_batch = self.style_image.repeat(batch_size, 1, 1, 1)
        return style_batch.to(device)

    def get_single(self, device="cpu"):
        """Get single style image"""
        return self.style_image.to(device)


def get_content_loader(
    content_dir, batch_size=4, image_size=256, num_workers=4, shuffle=True
):
    """
    Create DataLoader for content images

    Args:
        content_dir: Directory with content images
        batch_size: Batch size
        image_size: Image dimension
        num_workers: Number of worker threads
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    dataset = ContentDataset(content_dir, image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Ensure consistent batch sizes
    )

    return loader


def load_single_image(image_path, image_size=256, device="cpu"):
    """
    Load a single image for inference

    Args:
        image_path: Path to image
        image_size: Size to resize to
        device: Device to move tensor to

    Returns:
        image: (1, 3, H, W) tensor
    """
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)


def save_image(tensor, save_path):
    """
    Save tensor as image

    Args:
        tensor: (1, 3, H, W) or (3, H, W) in [0, 1]
        save_path: Output path
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Convert to PIL
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.cpu().clamp(0, 1))

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    image.save(save_path)
    print(f"Saved image to {save_path}")
