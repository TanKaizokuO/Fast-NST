# Fast Neural Style Transfer

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Transform your photos into stunning artworks in real-time using deep learning! This implementation of Fast Neural Style Transfer (Johnson et al.) allows you to apply artistic styles to any image in milliseconds.

---

## 🚀 Quick Start - Running the Web App

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### 1. Clone and Setup

```bash
# to Navigate to project directory
cd fast_nst

# create virtual environment using python or you can use conda
python -m venv venv

# activate the venv
Mac/Linux - source venv/bin/activate
Windows - venv\Scripts\activate 

# Install the dependencies
pip install -r requirements_web.txt
```

### 2. Download Model (Option A)

If you have a pre-trained model:

```bash
# Place your final_model.pth in the checkpoints folder
mkdir -p checkpoints
# Copy your model: cp /path/to/final_model.pth checkpoints/
```

### 3. Run the Web Application

```bash
# Start the Flask server
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

### 4. Stylize Images via Web Interface

1. **Upload** your photo (drag & drop or click)
2. **Adjust** image size (256px - 1024px)
3. **Click** "✨ Stylize Image"
4. **Download** your stylized result!

### Alternative: Command Line Stylization

```bash
# Stylize a single image
python stylize.py \
  --checkpoint checkpoints/final_model.pth \
  --input test_images/photo.jpg \
  --output stylized_photo.jpg \
  --image-size 512

# Stylize entire directory
python stylize.py \
  --checkpoint checkpoints/final_model.pth \
  --input test_images/ \
  --output stylized_outputs/
```

---

## 🎨 Training Your Own Style Model

### Step 1: Prepare Your Dataset

#### Content Images (Training Data)

**Option A: Download COCO Dataset (Recommended)**
```bash
# Create data directory
mkdir -p data/content_images

# Download COCO train2017 (~18GB, 118K images)
cd data
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
mv train2017/* content_images/
rm -rf train2017 train2017.zip

cd ..
```

**Option B: Use Your Own Images**
```bash
# Just copy your images into the folder
mkdir -p data/content_images
cp ~/Pictures/*.jpg data/content_images/

# You need at least 100+ diverse images for good results
```

#### Style Image

```bash
# Create styles directory
mkdir -p data/styles

# Download example style images
cd data/styles

# Van Gogh's Starry Night
wget "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg" -O starry_night.jpg

# Or use your own artwork
cp ~/path/to/your/artwork.jpg ./

cd ../..
```

### Step 2: Verify Setup with Diagnostic Tool

```bash
# Run diagnostic to check everything works
python diagnose.py \
  data/content_images/000000000009.jpg \
  data/styles/starry_night.jpg
```

Expected output:
```
✓ PyTorch installation OK
✓ Model architecture OK
✓ Output range looks good
✓ No NaN values
✓ Losses look reasonable
```

### Step 3: Train the Model

#### Quick Test (5 minutes - verify everything works)

```bash
python train.py \
  --content-dir ./data/content_images \
  --style-image ./data/styles/starry_night.jpg \
  --checkpoint-dir ./checkpoints/test \
  --output-dir ./outputs/test \
  --batch-size 2 \
  --max-steps 1000 \
  --style-weight 1e6 \
  --sample-interval 200
```

#### Full Training (2-4 hours with GPU)

```bash
python train.py \
  --content-dir ./data/content_images \
  --style-image ./data/styles/starry_night.jpg \
  --checkpoint-dir ./checkpoints/starry_night \
  --output-dir ./outputs/starry_night \
  --batch-size 4 \
  --max-steps 40000 \
  --style-weight 5e6 \
  --content-weight 1.0 \
  --tv-weight 1e-6 \
  --use-amp \
  --num-workers 4
```

#### Advanced Training Options

```bash
# For stronger stylization
python train.py \
  --content-dir ./data/content_images \
  --style-image ./data/styles/mosaic.jpg \
  --style-weight 1e7 \
  --max-steps 60000

# For multiple GPUs
python train.py \
  --batch-size 16 \
  --num-workers 8 \
  --use-amp

# Resume from checkpoint
python train.py \
  --resume ./checkpoints/starry_night/checkpoint_step_20000.pth \
  --content-dir ./data/content_images \
  --style-image ./data/styles/starry_night.jpg
```

### Step 4: Monitor Training Progress

Training will automatically save:
- **Sample outputs** → `outputs/starry_night/sample_step_XXXX.jpg`
- **Checkpoints** → `checkpoints/starry_night/checkpoint_step_XXXX.pth`
- **Final model** → `checkpoints/starry_night/final_model.pth`

**Expected Timeline:**
- **Step 500**: Subtle style beginning to show
- **Step 2,000**: Clear artistic effect visible
- **Step 10,000**: Strong stylization
- **Step 40,000**: Production-quality results ✨

**Monitor losses in terminal:**
```
Step 1000/40000
  Total Loss:   345678.12
  Content Loss: 23.45
  Style Loss:   3.45e+05
  TV Loss:      1.23e-04
```

### Step 5: Test Your Trained Model

```bash
# Stylize a test image
python stylize.py \
  --checkpoint ./checkpoints/starry_night/final_model.pth \
  --input test_photo.jpg \
  --output stylized_result.jpg
```

---

## 📁 Project Structure

```
fast_nst/
│
├── README.md                          # This file
├── QUICKSTART.md                      # Quick setup guide
├── WEB_DEPLOYMENT.md                  # Web deployment guide
├── DEBUGGING_GUIDE.md                 # Troubleshooting help
├── EMERGENCY_FIX.md                   # Common issues and fixes
│
├── requirements.txt                   # Training dependencies
├── requirements_web.txt               # Web app dependencies
│
├── Core Training Files
│   ├── models.py                      # TransformerNetwork architecture
│   ├── vgg_loss.py                    # VGG perceptual loss
│   ├── dataset.py                     # Data loading utilities
│   ├── train.py                       # Training script
│   ├── stylize.py                     # Inference script
│   ├── utils.py                       # Helper functions
│   └── diagnose.py                    # Diagnostic tool
│
├── Web Application
│   ├── app.py                         # Flask web server
│   └── templates/
│       └── index.html                 # Web interface
│
├── Data Structure
│   └── data/
│       ├── content_images/            # Training images (COCO/custom)
│       │   ├── 000000000001.jpg
│       │   ├── 000000000002.jpg
│       │   └── ...
│       └── styles/                    # Style images
│           ├── starry_night.jpg
│           ├── mosaic.jpg
│           └── wave.jpg
│
├── Training Outputs
│   ├── checkpoints/                   # Saved model weights
│   │   └── starry_night/
│   │       ├── checkpoint_step_2000.pth
│   │       ├── checkpoint_step_4000.pth
│   │       └── final_model.pth
│   └── outputs/                       # Sample stylized images during training
│       └── starry_night/
│           ├── style_reference.jpg
│           ├── sample_step_500.jpg
│           ├── content_step_500.jpg
│           └── ...
│
└── Web App Runtime (auto-created)
    ├── uploads/                       # Temporary uploads
    └── outputs/                       # Temporary outputs
```

---

## 🎯 Training Parameters Reference

| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `--batch-size` | 4 | Images per batch | 2-16 (depends on GPU) |
| `--image-size` | 256 | Training image size | 256-512 |
| `--lr` | 1e-3 | Learning rate | 1e-4 to 1e-3 |
| `--max-steps` | 40000 | Total training steps | 20000-80000 |
| `--style-weight` | 1e6 | Style loss importance | **1e6 to 1e7** ⚠️ |
| `--content-weight` | 1.0 | Content loss importance | 0.5-2.0 |
| `--tv-weight` | 1e-6 | Smoothness penalty | 1e-7 to 1e-5 |
| `--checkpoint-interval` | 2000 | Save frequency | 1000-5000 |
| `--sample-interval` | 500 | Sample output frequency | 200-1000 |
| `--use-amp` | False | Mixed precision training | Use for 2x speedup |

### ⚠️ Critical Parameter Warnings

**Style Weight:**
- ✅ **1e6 - 1e7**: Safe range, good results
- ⚠️ **1e8 - 1e9**: Very high, may cause issues
- ❌ **> 1e9**: Will cause glitchy outputs!

**If you see glitchy/corrupted outputs, your style weight is TOO HIGH!**

---

## 🏗️ Architecture Details

### Transformer Network (Generator)
- **Input/Output**: (B, 3, 256, 256)
- **Encoder**: 3 convolutional layers with Instance Normalization
- **Bottleneck**: 5 residual blocks (128 channels)
- **Decoder**: Upsample + Conv (no ConvTranspose2d)
- **Padding**: Reflection padding (prevents edge artifacts)
- **Output**: Tanh activation scaled to [0, 1]

### Loss Network (VGG19)
- **Content Loss**: MSE at `relu2_2`
- **Style Loss**: Gram matrix MSE at `relu1_2, relu2_2, relu3_3, relu4_3`
- **TV Loss**: Pixel smoothness penalty
- **Total Loss**: `L = λ_c·L_content + λ_s·L_style + λ_tv·L_tv`

### Key Implementation Features
✅ Instance Normalization (not Batch Normalization)  
✅ Reflection Padding (prevents boundary artifacts)  
✅ Nearest Neighbor Upsampling + Conv (no checkerboard)  
✅ Pre-cached Style Gram Matrices (efficiency)  
✅ Gradient Clipping (training stability)  
✅ Mixed Precision Training (2x faster)

---

## 💻 System Requirements

### Minimum Requirements (CPU Training)
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 20GB+
- Training time: ~24+ hours for 40K steps

### Recommended (GPU Training)
- GPU: NVIDIA RTX 3060 or better (6GB+ VRAM)
- RAM: 16GB+
- Storage: 50GB+ (for COCO dataset)
- Training time: ~2-4 hours for 40K steps

### Optimal (Fast Training)
- GPU: NVIDIA RTX 3090 / A100 (24GB+ VRAM)
- RAM: 32GB+
- Batch size: 8-16
- Training time: ~1-2 hours for 40K steps

---

## 🐛 Troubleshooting

### Training Issues

**"CUDA out of memory"**
```bash
# Reduce batch size
python train.py --batch-size 2 ...

# Or reduce image size
python train.py --image-size 128 ...

# Or use CPU
python train.py --cpu ...
```

**Glitchy/corrupted outputs**
```bash
# Your style weight is TOO HIGH!
# Use 1e6 instead of 5e10
python train.py --style-weight 1e6 ...

# Run diagnostic first
python diagnose.py data/content_images/test.jpg data/styles/style.jpg
```

**Training very slow**
```bash
# Enable mixed precision (requires GPU)
python train.py --use-amp ...

# Increase batch size (if you have GPU memory)
python train.py --batch-size 8 ...

# Use more workers
python train.py --num-workers 8 ...
```

**Loss becomes NaN**
```bash
# Reduce learning rate
python train.py --lr 5e-4 ...

# Reduce style weight
python train.py --style-weight 5e5 ...
```

### Web App Issues

**"Module not found: Flask"**
```bash
pip install Flask Werkzeug
```

**"Model checkpoint not found"**
```bash
# Update path in app.py line 19
CHECKPOINT_PATH = 'checkpoints/starry_night/final_model.pth'
```

**Port already in use**
```bash
# Change port in app.py (last line)
app.run(debug=True, host='0.0.0.0', port=8000)
```

**Stylization fails in web app**
```bash
# Check terminal for error messages
# Common: Model architecture mismatch - update models.py
```

---

## 📊 Performance Benchmarks

| GPU Model | Batch Size | Training Speed | Inference Speed |
|-----------|------------|----------------|-----------------|
| RTX 4090 | 16 | ~1 hour | ~5ms/image |
| RTX 3090 | 8 | ~1.5 hours | ~10ms/image |
| RTX 3060 | 4 | ~3 hours | ~20ms/image |
| GTX 1080 | 2 | ~5 hours | ~30ms/image |
| CPU (i7) | 1 | ~24+ hours | ~500ms/image |

*Benchmarks for 40,000 training steps on 256x256 images*

---

## 🎨 Example Results

Train different models for different artistic styles:

```bash
# Van Gogh's Starry Night
python train.py --style-image data/styles/starry_night.jpg --style-weight 5e6

# Picasso's Cubist style
python train.py --style-image data/styles/picasso.jpg --style-weight 1e7

# Japanese Wave (Hokusai)
python train.py --style-image data/styles/wave.jpg --style-weight 5e6

# Mosaic/Stained Glass
python train.py --style-image data/styles/mosaic.jpg --style-weight 1e7
```

---

## 📚 Additional Resources

### Documentation
- **QUICKSTART.md** - Fast setup guide
- **WEB_DEPLOYMENT.md** - Deploy to cloud (Heroku, Railway, Replicate)
- **DEBUGGING_GUIDE.md** - Detailed troubleshooting
- **EMERGENCY_FIX.md** - Quick fixes for common issues

### Research Papers
- Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", ECCV 2016
- Ulyanov et al., "Instance Normalization: The Missing Ingredient for Fast Stylization", 2016
- Gatys et al., "A Neural Algorithm of Artistic Style", 2015

### Related Projects
- Original implementation: https://github.com/jcjohnson/fast-neural-style
- PyTorch examples: https://github.com/pytorch/examples/tree/master/fast_neural_style

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional style transfer architectures (MSG-Net, AdaIN)
- Video style transfer support
- Mobile optimization (TorchScript, ONNX)
- Batch processing API
- Additional loss functions

---

## 📄 License

MIT License - Feel free to use for personal or commercial projects.

---

## 🙏 Acknowledgments

- Original Fast Neural Style Transfer paper by Johnson et al.
- PyTorch team for the excellent framework
- COCO dataset for training images
- All the artists whose work inspires these styles

---

## 📧 Support

If you encounter issues:

1. Check **DEBUGGING_GUIDE.md** for common problems
2. Run `python diagnose.py` to identify issues
3. Verify your parameters (especially style weight!)
4. Check that your GPU drivers are up to date

**Common Issues:**
- Glitchy outputs → Style weight too high (use 1e6, not 5e10!)
- Out of memory → Reduce batch size or image size
- Slow training → Enable `--use-amp` and increase `--num-workers`

---

## 🌟 Quick Command Reference

```bash
# Train model
python train.py --content-dir ./data/content_images --style-image ./data/styles/starry_night.jpg --max-steps 40000 --use-amp

# Stylize image
python stylize.py --checkpoint checkpoints/final_model.pth --input photo.jpg --output stylized.jpg

# Run web app
python app.py

# Diagnose issues
python diagnose.py data/content_images/test.jpg data/styles/style.jpg
```

---

**Happy Stylizing! 🎨✨**

