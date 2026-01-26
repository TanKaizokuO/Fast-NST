from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import io
import os
from pathlib import Path
import base64

from models import TransformerNetwork
from dataset import save_image
import torchvision.transforms as transforms

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"

# Create directories
Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)
Path(app.config["OUTPUT_FOLDER"]).mkdir(exist_ok=True)

# Load model
CHECKPOINT_PATH = "models/Crazy.pth"

# Check if model exists
if not os.path.exists(CHECKPOINT_PATH):
    print(f"ERROR: Model checkpoint not found at {CHECKPOINT_PATH}")
    print("Please ensure final_model.pth is in the same directory as app.py")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNetwork().to(device)

# Load checkpoint
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Model loaded from {CHECKPOINT_PATH}")
    print(f"✓ Using device: {device}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    exit(1)

# Image transform
transform = transforms.Compose([transforms.ToTensor()])
to_pil = transforms.ToPILImage()


def stylize_image(image, image_size=512):
    """Stylize a PIL image"""
    # Resize while maintaining aspect ratio
    width, height = image.size
    if max(width, height) > image_size:
        if width > height:
            new_width = image_size
            new_height = int(height * (image_size / width))
        else:
            new_height = image_size
            new_width = int(width * (image_size / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)

    # Convert to tensor
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Stylize
    with torch.no_grad():
        stylized_tensor = model(img_tensor)

    # Convert back to PIL
    stylized_pil = to_pil(stylized_tensor.squeeze(0).cpu().clamp(0, 1))

    return stylized_pil


@app.route("/")
def index():
    """Home page"""
    return render_template("index.html")


@app.route("/stylize", methods=["POST"])
def stylize():
    """API endpoint to stylize an image"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Read image
        image = Image.open(file.stream).convert("RGB")

        # Get image size parameter
        image_size = int(request.form.get("size", 512))
        image_size = min(max(image_size, 256), 1024)  # Clamp to 256-1024

        print(f"Processing image: {file.filename}, size: {image_size}")

        # Stylize
        stylized = stylize_image(image, image_size)

        # Convert to bytes
        img_io = io.BytesIO()
        stylized.save(img_io, "JPEG", quality=95)
        img_io.seek(0)

        # Encode to base64 for JSON response
        img_base64 = base64.b64encode(img_io.getvalue()).decode()

        print(f"✓ Image stylized successfully")

        return jsonify(
            {"success": True, "image": f"data:image/jpeg;base64,{img_base64}"}
        )

    except Exception as e:
        print(f"ERROR in stylize endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/stylize_download", methods=["POST"])
def stylize_download():
    """Stylize and return downloadable image"""
    try:
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]

        if file.filename == "":
            return "No file selected", 400

        # Read image
        image = Image.open(file.stream).convert("RGB")

        # Stylize
        image_size = int(request.form.get("size", 512))
        image_size = min(max(image_size, 256), 1024)
        stylized = stylize_image(image, image_size)

        # Save to bytes
        img_io = io.BytesIO()
        stylized.save(img_io, "JPEG", quality=95)
        img_io.seek(0)

        return send_file(
            img_io,
            mimetype="image/jpeg",
            as_attachment=True,
            download_name="stylized_image.jpg",
        )

    except Exception as e:
        print(f"ERROR in stylize_download: {str(e)}")
        import traceback
        traceback.print_exc()
        return str(e), 500


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎨 Neural Style Transfer Server Starting...")
    print("="*50)
    
    # Get port from environment variable (Render assigns this dynamically)
    port = int(os.environ.get("PORT", 5000))
    
    print(f"Server will run on port: {port}")
    print("="*50 + "\n")
    
    # Bind to 0.0.0.0 to accept external connections
    app.run(debug=False, host="0.0.0.0", port=port)