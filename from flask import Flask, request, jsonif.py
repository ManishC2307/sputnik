from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Allows your phone browser to talk to this server

# Load your 3.64 KB TinyML model
interpreter = tf.lite.Interpreter(model_path="hazard_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

HAZARD_CLASSES = {
    0: "Normal Environmental Conditions - No Hazard",
    1: "Extreme Heat Stress Risk - Hydrate and seek shade!",
    2: "Hazardous Air Quality / Toxic Gas Detected - Wear mask!",
    3: "Severe Storm / Low Pressure Alert - Take cover!"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Expected input array format: [temp, humidity, pressure, gas, wind, sound, dust]
        telemetry = np.array(data['telemetry'], dtype=np.float32).reshape(1, 7)
        
        interpreter.set_tensor(input_details[0]['index'], telemetry)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        return jsonify({
            "success": True,
            "hazard_class": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "warning_message": HAZARD_CLASSES.get(predicted_class, "Unknown Hazard")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    # Running on 0.0.0.0 lets any device on your Wi-Fi connect
    app.run(host='0.0.0.0', port=5000, debug=True)