"""
Inference script for stylizing images with trained model
"""

import torch
import argparse
from pathlib import Path
import time

from models import TransformerNetwork
from dataset import load_single_image, save_image
from utils import load_checkpoint, get_device


def stylize_image(model, image_path, output_path, image_size=256, device="cpu"):
    """
    Stylize a single image

    Args:
        model: Trained TransformerNetwork
        image_path: Path to input image
        output_path: Path to save stylized output
        image_size: Size to process image at
        device: Device to run on
    """
    model.eval()

    # Load image
    print(f"Loading image from {image_path}...")
    image = load_single_image(image_path, image_size, device)

    # Stylize
    print("Stylizing...")
    start_time = time.time()

    with torch.no_grad():
        stylized = model(image)

    elapsed = time.time() - start_time
    print(f"Stylization completed in {elapsed:.3f} seconds")

    # Save
    save_image(stylized, output_path)

    return stylized


def stylize_batch(model, input_dir, output_dir, image_size=256, device="cpu"):
    """
    Stylize all images in a directory

    Args:
        model: Trained TransformerNetwork
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        image_size: Size to process images at
        device: Device to run on
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = []

    for ext in valid_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))

    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to stylize")

    # Process each image
    for img_path in image_files:
        output_path = output_dir / f"stylized_{img_path.name}"
        print(f"\nProcessing {img_path.name}...")

        try:
            stylize_image(model, img_path, output_path, image_size, device)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"\nAll images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stylize images with trained model")

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input image or directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output image or directory"
    )

    # Optional arguments
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Size to process images (default: 256)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage even if GPU available"
    )

    args = parser.parse_args()

    # Setup device
    if args.cpu:
        device = torch.device("cpu")
        print("Using CPU (forced)")
    else:
        device = get_device()

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = TransformerNetwork().to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    # Check if input is file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image
        stylize_image(model, args.input, args.output, args.image_size, device)
    elif input_path.is_dir():
        # Directory of images
        stylize_batch(model, args.input, args.output, args.image_size, device)
    else:
        print(f"Error: {args.input} is neither a file nor directory")
        return

    print("\nDone!")


if __name__ == "__main__":
    main()
