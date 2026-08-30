from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import requests

app = Flask(__name__)
CORS(app)

# 1. Load TinyML Hazard Model
interpreter = tf.lite.Interpreter(model_path="hazard_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

FEATURE_MINS = np.array([10.0, 0.0, 950.0, 0.0, 40.0, 70.0, 5.0], dtype=np.float32)
FEATURE_MAXS = np.array([50.0, 100.0, 1030.0, 100.0, 180.0, 100.0, 40.0], dtype=np.float32)

def normalize_features(raw_data):
    return (raw_data - FEATURE_MINS) / (FEATURE_MAXS - FEATURE_MINS)

HAZARD_CLASSES = {
    0: "Status Normal: Environmental and physiological telemetry within safe bounds.",
    1: "Early Warning: Extreme Heat Stress Risk detected. High probability of heat exhaustion—hydrate immediately and move to shade.",
    2: "Early Warning: Flash Flood / Heavy Rain anomaly detected in your micro-zone. Move to higher ground.",
    3: "Early Warning: Hazardous PM2.5 Pollution Spike. Respiratory stress detected—wear an N95 mask or head indoors."
}

# --- ROUTE 1: TinyML Hazard Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        raw_telemetry = np.array(data['telemetry'], dtype=np.float32).reshape(1, 7)
        normalized_telemetry = normalize_features(raw_telemetry)
        
        interpreter.set_tensor(input_details[0]['index'], normalized_telemetry)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100.0

        return jsonify({
            "success": True,
            "hazard_class": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "warning_message": HAZARD_CLASSES.get(predicted_class, "Unknown Hazard Class")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# --- ROUTE 2: Offline Medical Assistant (Ollama LLM) ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_msg = request.json.get('message', '')
        
        # System prompt forces medical grounding based on WHO protocols
        system_prompt = (
            "You are Team Sputnik's offline emergency first-aid assistant. "
            "Provide concise, direct, step-by-step emergency guidance based on WHO protocols. "
            "Keep answers strictly under 3 short sentences."
        )

        # Connect locally to Ollama running on port 11434
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "llama3.2:1b",
                "prompt": f"{system_prompt}\n\nUser Question: {user_msg}\nAnswer:",
                "stream": False
            },
            timeout=30
        )
        
        bot_reply = response.json().get('response', 'No response generated.')
        return jsonify({"success": True, "reply": bot_reply})

    except Exception as e:
        return jsonify({"success": False, "error": "Make sure Ollama is running in background! " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)