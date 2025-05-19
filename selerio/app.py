import os
import json
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'skin_cancer_model')

# Initialize Flask app
app = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
def load_api_key():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        load_dotenv()
        api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    genai.configure(api_key=api_key)
    return api_key

# Setup directories
def setup_directories():
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

# Load TFLite model
def load_model():
    model_path = os.path.join(MODEL_DIR, 'model.tflite')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

# Load labels
def load_labels():
    labels_path = os.path.join(MODEL_DIR, 'labels.txt')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    with open(labels_path, 'r') as f:
        return f.read().strip().split('\n')

# Load label names
def load_label_names():
    labels_json_path = os.path.join(MODEL_DIR, 'labels.json')
    if not os.path.exists(labels_json_path):
        raise FileNotFoundError(f"Labels JSON file not found at {labels_json_path}")
    with open(labels_json_path, 'r') as f:
        label_data = json.load(f)
        return {item['label']: item['name'] for item in label_data}

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0).astype(np.float32)

# Initialize system
API_KEY = load_api_key()
setup_directories()
interpreter, input_details, output_details = load_model()
labels = load_labels()
label_names = load_label_names()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'example' in request.form:
            file_path = os.path.join(BASE_DIR, 'static', 'images', request.form['example'])
            if not os.path.exists(file_path):
                return jsonify({'error': f'Example image {request.form["example"]} not found'}), 404
        else:
            if 'file' not in request.files or not request.files['file'].filename:
                return jsonify({'error': 'No file uploaded'}), 400

            file = request.files['file']
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400

            file_path = os.path.join('static/uploads', secure_filename(file.filename))
            file.save(file_path)

        input_data = preprocess_image(file_path)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = np.argmax(predictions)
        predicted_label = labels[predicted_index]
        if predicted_label not in label_names:
            return jsonify({'error': f'Unknown label: {predicted_label}'}), 500

        predicted_name = label_names[predicted_label]
        confidence = float(predictions[predicted_index])

        if 'file' in request.files and os.path.exists(file_path):
            os.remove(file_path)

        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f'Provide a detailed description and recommended treatment for {predicted_name} in 8 sentences.'
        response = model.generate_content(prompt)
        description = response.text

        return jsonify({
            'label': predicted_label,
            'name': predicted_name,
            'confidence': confidence,
            'description_and_treatment': description
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
