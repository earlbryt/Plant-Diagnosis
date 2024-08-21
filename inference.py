from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
import io
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Load YOLO model
model = YOLO('best.pt')  # Replace with your model path

# Load Poppins font
font_path = "Poppins-Regular.ttf"  # Make sure this path is correct
font_size = 20

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Read image file
    image_stream = io.BytesIO(file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run inference
    results = model(img)

    # Convert OpenCV image to PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Load Poppins font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}. Using default font.")
        font = ImageFont.load_default()

    # Process results and draw bounding boxes
    for result in results:
        boxes = result.boxes  # Assuming the result object has a 'boxes' attribute
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Assuming xyxy format
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            draw.rectangle([x1, y1, x2, y2], outline=(191, 64, 191), width=3)
            
            # Add label with background patch
            label = f"{result.names[int(box.cls[0])]}"
            
            # Get text size
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Add padding
            padding = 4
            patch_width = text_width + 2 * padding
            patch_height = text_height + 4 * padding
            
            # Calculate position for the label patch (on top of the bounding box)
            patch_x1 = x1
            patch_y1 = y1 - patch_height  # Move the patch up by its height
            
            # Ensure the patch doesn't go above the image boundary
            if patch_y1 < 0:
                patch_y1 = 0
            
            # Create background patch
            draw.rectangle([patch_x1, patch_y1, patch_x1 + patch_width, patch_y1 + patch_height], fill=(191, 64, 191))
            
            # Put text on patch
            draw.text((patch_x1 + padding, patch_y1 + padding), label, font=font, fill=(255, 255, 255))

    # Convert PIL Image back to OpenCV format
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Encode the image to send back
    _, img_encoded = cv2.imencode('.png', img)
    return send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/png',
        as_attachment=False
    )

if __name__ == '__main__':
    app.run(debug=True)