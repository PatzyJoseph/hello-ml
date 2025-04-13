import os
import json
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables or .env file.")

# ✅ Configure Gemini API here
genai.configure(api_key=gemini_api_key)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
LABELS_FILE = os.path.join(BASE_DIR, "models/labels.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_unquant.tflite")

# Initialize Flask app
app = Flask(__name__)

# Load TensorFlow Lite model
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info("TFLite model loaded successfully.")
        return interpreter, input_details, output_details
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the model")

# Initialize the TFLite model interpreter and details
interpreter, input_details, output_details = load_model()

# Read labels from JSON 
def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            label_data = json.load(f)
        return [entry["name"] for entry in label_data]
    else:
        logger.error("labels.json file not found")
        return []

def preprocess_image(image_path, input_shape):
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((input_shape[1], input_shape[2]))
        image_array = np.array(image, dtype=np.float32)
        image_array /= 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.info("Image preprocessed successfully.")
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise RuntimeError("Failed to preprocess the image")

def details(disease_name):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            contents=[
                f"Provide a concise 3-sentence description of {disease_name}, including causes, risk factors, and possible treatments."
            ]
        )
        return response.text.strip() if response else "No AI-generated details available."
    except Exception as e:
        logger.error(f"Error generating AI details: {e}")
        return "Error generating AI details."
    
    # Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part in the request")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    try:
        # Preprocess the image
        input_shape = input_details[0]['shape']
        image_array = preprocess_image(file, input_shape)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(output_data[0])
        confidence = output_data[0][predicted_class]
        
        labels = load_labels()
        predicted_label = labels[predicted_class] if labels else "Unknown"
        disease_name = predicted_label  # Assign the predicted label to disease_name
        
        # Generate AI details
        ai_details = details(disease_name)
        
        # Render the index.html template with results
        return render_template('index.html', 
                               label_name=predicted_label, 
                               disease_name=disease_name, 
                               confidence=confidence, 
                               ai_details=ai_details)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)