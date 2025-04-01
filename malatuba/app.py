import json
import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google import genai

# Load environment variables
load_dotenv('.env')
api_key = os.environ.get('GENAI_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Google GenAI client
client = genai.Client(api_key=api_key)


def load_model():
    """Loads the TensorFlow Lite model."""
    try:
        model_path = "skin-cancer-classifier-starter-main/skin_cancer_model/model.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model")


# Load model once at startup
interpreter, input_details, output_details = load_model()


def load_labels():
    """Loads the labels for class prediction."""
    labels_path = "skin-cancer-classifier-starter-main/skin_cancer_model/labels.json"
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r") as f:
                label_data = json.load(f)
            return [entry.get("name", "Unknown") for entry in label_data]  # Safeguard for missing keys
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return []
    else:
        logger.error("labels.json file not found")
        return []


def process_image(image_path):
    """Preprocesses the image for model inference."""
    try:
        with Image.open(image_path) as img:  # Using 'with' ensures proper file closing
            img = img.convert("RGB")
            img = img.resize((224, 224))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        return img
    except UnidentifiedImageError:
        logger.error("Invalid image format")
        raise ValueError("Invalid image format")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise


def generate_cancer_details(disease_name):
    """Uses Gemini AI to generate information about the detected skin condition."""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                f"Provide a concise 3-sentence description of {disease_name}, including causes, risk factors, and possible treatments."
            ]
        )
        return response.candidates[0].content.parts[0].text.strip() if response.candidates else "No AI-generated details available."
    except Exception as e:
        logger.error(f"Error generating AI details: {e}")
        return "Error generating AI details."


@app.route("/", methods=["GET"])
def home():
    """Renders the homepage."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    """Handles image uploads and runs inference."""
    logger.info("Received request to upload image")

    if "image" not in request.files:
        logger.error("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(BASE_DIR, "uploads", file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        logger.info(f"Image saved to {file_path}")

        img = None  # Initialize to avoid reference issues
        try:
            img = process_image(file_path).astype(np.float32)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({"error": "Failed to process image"}), 400

        # Ensure model is loaded
        if interpreter is None:
            logger.error("Model is not loaded")
            return jsonify({"error": "Model is not loaded"}), 500

        # Perform inference
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])

        predicted_class_index = np.argmax(predictions)
        confidence_score = float(predictions[0][predicted_class_index])

        class_labels = load_labels()
        predicted_class = class_labels[predicted_class_index] if predicted_class_index < len(class_labels) else "Unknown"

        ai_details = generate_cancer_details(predicted_class)

        response = {
            "prediction": predicted_class,
            "confidence": confidence_score,
            "ai_generated_info": ai_details
        }

        # Ensure file deletion after processing
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file: {e}")

        return jsonify(response)

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
