"""
Training script for Fast Neural Style Transfer
"""

import time
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
from pathlib import Path

from models import TransformerNetwork
from vgg_loss import StyleTransferLoss
from dataset import get_content_loader, StyleImageLoader, save_image
from utils import (
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    get_device,
    count_parameters,
    save_config,
)


def train(args):
    """Main training loop"""

    # Setup device
    device = get_device()

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    save_config(config, Path(args.checkpoint_dir) / "config.json")

    # Initialize model
    print("\nInitializing model...")
    model = TransformerNetwork().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Initialize loss function
    print("\nInitializing loss function...")
    criterion = StyleTransferLoss(
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
    ).to(device)

    # Load and cache style image
    print("\nLoading style image...")
    style_loader = StyleImageLoader(args.style_image, image_size=args.image_size)
    style_img = style_loader.get_single(device)
    criterion.set_style_image(style_img)

    # Save style image for reference
    save_image(style_img, Path(args.output_dir) / "style_reference.jpg")

    # Initialize data loader
    print("\nInitializing data loader...")
    content_loader = get_content_loader(
        args.content_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None

    # Resume from checkpoint if exists
    start_step = 0
    if args.resume and Path(args.resume).exists():
        start_step = load_checkpoint(model, args.resume, optimizer, device)

    # Training loop
    print("\n" + "=" * 50)
    print("Starting training...")
    print(f"Style weight: {args.style_weight}")
    print(f"Content weight: {args.content_weight}")
    print(f"TV weight: {args.tv_weight}")
    print("=" * 50 + "\n")

    model.train()

    # Meters for tracking
    total_loss_meter = AverageMeter()
    content_loss_meter = AverageMeter()
    style_loss_meter = AverageMeter()
    tv_loss_meter = AverageMeter()

    # Sanity check weights
    if args.style_weight > 1e8:
        print(f"WARNING: style_weight={args.style_weight:.2e} is extremely high!")
        print("Recommended: 1e6 to 1e7")
        print("Continue anyway? (Ctrl+C to cancel)")
        import time

        time.sleep(5)

    step = start_step
    epoch = 0

    while step < args.max_steps:
        epoch += 1

        pbar = tqdm(content_loader, desc=f"Epoch {epoch}")

        for batch_idx, content in enumerate(pbar):
            content = content.to(device)

            # Forward pass
            if args.use_amp:
                with autocast():
                    generated = model(content)
                    total_loss, c_loss, s_loss, tv_loss = criterion(generated, content)
            else:
                generated = model(content)
                total_loss, c_loss, s_loss, tv_loss = criterion(generated, content)

            # Backward pass
            optimizer.zero_grad()

            if args.use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # Update meters
            total_loss_meter.update(total_loss.item())
            content_loss_meter.update(c_loss.item())
            style_loss_meter.update(s_loss.item())
            tv_loss_meter.update(tv_loss.item())

            # Update progress bar
            pbar.set_postfix(
                {
                    "total": f"{total_loss_meter.avg:.2f}",
                    "content": f"{content_loss_meter.avg:.2f}",
                    "style": f"{style_loss_meter.avg:.2e}",
                    "tv": f"{tv_loss_meter.avg:.2e}",
                }
            )

            step += 1

            # Save checkpoint
            if step % args.checkpoint_interval == 0:
                checkpoint_path = (
                    Path(args.checkpoint_dir) / f"checkpoint_step_{step}.pth"
                )
                save_checkpoint(
                    model, optimizer, step, total_loss.item(), checkpoint_path
                )

            # Save sample outputs
            if step % args.sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    sample_output = model(content[:1])
                    save_image(
                        sample_output, Path(args.output_dir) / f"sample_step_{step}.jpg"
                    )
                    save_image(
                        content[:1], Path(args.output_dir) / f"content_step_{step}.jpg"
                    )
                model.train()

            # Print detailed statistics
            if step % args.log_interval == 0:
                print(f"\nStep {step}/{args.max_steps}")
                print(f"  Total Loss:   {total_loss_meter.avg:.4f}")
                print(f"  Content Loss: {content_loss_meter.avg:.4f}")
                print(f"  Style Loss:   {style_loss_meter.avg:.2e}")
                print(f"  TV Loss:      {tv_loss_meter.avg:.2e}")

                # Reset meters
                total_loss_meter.reset()
                content_loss_meter.reset()
                style_loss_meter.reset()
                tv_loss_meter.reset()

            if step >= args.max_steps:
                break

        if step >= args.max_steps:
            break

    # Save final model
    final_path = Path(args.checkpoint_dir) / "final_model.pth"
    save_checkpoint(model, optimizer, step, total_loss.item(), final_path)

    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


def main():
    start_time = time.perf_counter()
    print(f"The start time is {start_time} seconds")
    parser = argparse.ArgumentParser(description="Train Fast Neural Style Transfer")

    # Paths
    parser.add_argument(
        "--content-dir",
        type=str,
        required=True,
        help="Directory containing content images (e.g., COCO)",
    )
    parser.add_argument(
        "--style-image", type=str, required=True, help="Path to style image"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save sample outputs",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="Size of training images"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--max-steps", type=int, default=40000, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Loss weights
    parser.add_argument(
        "--content-weight", type=float, default=1.0, help="Weight for content loss"
    )
    parser.add_argument(
        "--style-weight", type=float, default=1e6, help="Weight for style loss"
    )
    parser.add_argument(
        "--tv-weight", type=float, default=1e-6, help="Weight for total variation loss"
    )

    # Logging and saving
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=500,
        help="Save sample output every N steps",
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Print statistics every N steps"
    )

    # Mixed precision
    parser.add_argument(
        "--use-amp", action="store_true", help="Use automatic mixed precision training"
    )

    args = parser.parse_args()

    train(args)
    end_time = time.perf_counter()
    print(f"The end time is {end_time} seconds")
    execution_time = end_time - start_time
    print(f"The code executed in {execution_time} seconds")


# Code you want to measure goes here
# ...
if __name__ == "__main__":
    main()
