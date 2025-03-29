import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite")

from flask import Flask, request, render_template, jsonify, send_from_directory
import tensorflow.lite as tflite
import numpy as np
import cv2
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload folder exists

# Get absolute paths for model and labels
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labels_path = os.path.join(BASE_DIR, "skin_cancer_model", "labels.json")
model_path = os.path.join(BASE_DIR, "skin_cancer_model", "model.tflite")

# Debug: Ensure file paths exist
print(f"Looking for labels.json at: {labels_path}")
print(f"Labels.json exists: {os.path.exists(labels_path)}")
print(f"Looking for model.tflite at: {model_path}")
print(f"Model.tflite exists: {os.path.exists(model_path)}")

# Load labels
try:
    with open(labels_path, "r") as f:
        labels = json.load(f)
        label_dict = {item["label"]: item["name"] for item in labels}
except FileNotFoundError:
    print("Error: labels.json not found!")
    label_dict = {}

# Load TFLite model
try:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]  # Expected input size
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    input_shape = (224, 224)  # Default size if model load fails

def preprocess_image(image_path):
    """Prepares image for model input."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Configure Gemini AI with API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_disease_info(disease_name):
    """Fetches disease information using Gemini AI."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(f"Provide a short description of {disease_name}.")
        return response.text if response else "No additional information available."
    except Exception as e:
        print(f"Error fetching Gemini AI data: {e}")
        return "AI service unavailable."

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Handles image classification requests."""
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No file selected')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess and classify image
    input_data = preprocess_image(filepath)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)

    # Map label to disease name
    label_list = list(label_dict.keys())
    predicted_class = label_list[predicted_label] if predicted_label < len(label_list) else "Unknown"
    disease_name = label_dict.get(predicted_class, "Unknown")

    # Get disease info from Gemini AI
    disease_info = get_disease_info(disease_name)

    return render_template('index.html', disease=disease_name, info=disease_info, image=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves uploaded images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/gemini-info', methods=['GET'])
def gemini_info():
    """Fetches disease info via API."""
    disease_name = request.args.get('disease', '')
    if not disease_name:
        return jsonify({"error": "No disease specified"}), 400

    disease_info = get_disease_info(disease_name)
    return jsonify({"disease": disease_name, "info": disease_info})

if __name__ == '__main__':
    app.run(debug=True)
