from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from flask_cors import CORS
import tempfile
import os
import cv2

# Import preprocessing function
from preprocess import remove_background_to_black

PORT = int(os.getenv('PORT', 5000))

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load trained model
MODEL_PATH = "best_custom_cnn_model.h5"
model = load_model(MODEL_PATH)

# Define image size (depends on your ResNet50 input size, usually 224x224)
IMG_SIZE = (224, 224)

# Example class labels (replace with your actual classes)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file is present
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]

        # Save to a temporary location
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        file.save(temp_input.name)

        # Preprocess the image using your custom preprocessing script
        processed_img, _ = remove_background_to_black(temp_input.name)

        # Resize to match model input size
        processed_img = cv2.resize(processed_img, IMG_SIZE)

        # Normalize and prepare for model
        img_array = img_to_array(processed_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        preds = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(preds[0])]
        confidence = float(np.max(preds[0]))

        # Cleanup temp file
        # os.remove(temp_input.name)

        # Return result
        return jsonify({
            # "preds" : np.argmax(preds[0]),
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Plant Disease Detection API is running!"})


if __name__ == "__main__":
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', debug=debug)