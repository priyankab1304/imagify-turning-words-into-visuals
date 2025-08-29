from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
from PIL import Image
import io
import os
import base64
import logging
import time

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder="static")

# Configuration
HF_API_KEY = "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"
MODEL = "stabilityai/stable-diffusion-3.5-large"
MAX_RETRIES = 3
RETRY_DELAY = 1
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 1024

# Initialize client
client = InferenceClient(token=HF_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    start_time = time.time()
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Get parameters with defaults
    seed = int(data.get("seed", 0))
    width = int(data.get("width", DEFAULT_WIDTH))
    height = int(data.get("height", DEFAULT_HEIGHT))
    guidance_scale = float(data.get("guidanceScale", 4.5))
    steps = int(data.get("steps", 40))

    # Validate parameters
    width = max(256, min(1920, width))
    height = max(256, min(1920, height))
    guidance_scale = max(1.0, min(10.0, guidance_scale))
    steps = max(10, min(100, steps))

    try:
        for attempt in range(MAX_RETRIES):
            try:
                # Generate image with all parameters
                image_bytes = client.text_to_image(
                    prompt,
                    model=MODEL,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    seed=seed
                )
                
                # Process the image
                if isinstance(image_bytes, bytes):
                    image = Image.open(io.BytesIO(image_bytes))
                elif isinstance(image_bytes, Image.Image):
                    image = image_bytes
                else:
                    raise ValueError("Unexpected response format")
                
                if image.mode in ('RGBA', 'LA'):
                    image = image.convert('RGB')
                
                # Save to memory buffer
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                buffer.seek(0)
                
                # Create response
                encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{encoded_image}"
                
                logging.info(f"Image generated in {time.time() - start_time:.2f} seconds")
                return jsonify({
                    "success": True, 
                    "image_url": image_url,
                    "generation_time": f"{time.time() - start_time:.2f}s",
                    "dimensions": f"{width}x{height}",
                    "parameters": {
                        "seed": seed,
                        "width": width,
                        "height": height,
                        "guidance_scale": guidance_scale,
                        "steps": steps
                    }
                })

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                time.sleep(RETRY_DELAY * (attempt + 1))

    except Exception as e:
        logging.error("Error generating image", exc_info=True)
        return jsonify({
            "error": f"Error generating image: {str(e)}",
            "details": "Please try again with different parameters or check back later."
        }), 500

if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
