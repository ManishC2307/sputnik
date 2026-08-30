import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. SET RANDOM SEEDS FOR REPRODUCIBILITY
np.random.seed(42)
tf.random.set_seed(42)

# 2. FEATURE NORMALIZATION BOUNDS (For On-Device Edge Deployment)
# Feature order: [Temp_C, Humidity_%, Barometer_hPa, HRV_RMSSD_ms, HeartRate_BPM, SpO2_%, Respiration_PM]
FEATURE_MINS = np.array([10.0, 0.0, 950.0, 0.0, 40.0, 70.0, 5.0], dtype=np.float32)
FEATURE_MAXS = np.array([50.0, 100.0, 1030.0, 100.0, 180.0, 100.0, 40.0], dtype=np.float32)

def normalize_features(raw_data):
    """Min-Max scaling to range [0, 1] using fixed hardware bounds."""
    return (raw_data - FEATURE_MINS) / (FEATURE_MAXS - FEATURE_MINS)

# 3. SYNTHETIC MULTI-HAZARD TELEMETRY GENERATOR
def generate_synthetic_dataset(samples_per_class=1500):
    # Class 0: Normal Baseline Conditions
    c0_temp = np.random.uniform(20.0, 32.0, samples_per_class)
    c0_hum = np.random.uniform(30.0, 60.0, samples_per_class)
    c0_baro = np.random.uniform(1010.0, 1020.0, samples_per_class)
    c0_hrv = np.random.uniform(45.0, 85.0, samples_per_class)
    c0_hr = np.random.uniform(60.0, 80.0, samples_per_class)
    c0_spo2 = np.random.uniform(97.0, 100.0, samples_per_class)
    c0_resp = np.random.uniform(12.0, 18.0, samples_per_class)
    c0 = np.column_stack([c0_temp, c0_hum, c0_baro, c0_hrv, c0_hr, c0_spo2, c0_resp])
    y0 = np.zeros(samples_per_class, dtype=np.int32)

    # Class 1: Extreme Heat Stress (Heat Wave Alert)
    c1_temp = np.random.uniform(39.0, 48.0, samples_per_class)
    c1_hum = np.random.uniform(55.0, 90.0, samples_per_class)
    c1_baro = np.random.uniform(1005.0, 1015.0, samples_per_class)
    c1_hrv = np.random.uniform(10.0, 28.0, samples_per_class) # Significant HRV drop under thermal stress
    c1_hr = np.random.uniform(95.0, 135.0, samples_per_class)   # Tachycardia baseline spike
    c1_spo2 = np.random.uniform(94.0, 98.0, samples_per_class)
    c1_resp = np.random.uniform(18.0, 26.0, samples_per_class)
    c1 = np.column_stack([c1_temp, c1_hum, c1_baro, c1_hrv, c1_hr, c1_spo2, c1_resp])
    y1 = np.ones(samples_per_class, dtype=np.int32)

    # Class 2: Flash Flood / Heavy Rain Anomaly
    c2_temp = np.random.uniform(22.0, 29.0, samples_per_class)
    c2_hum = np.random.uniform(85.0, 100.0, samples_per_class)
    c2_baro = np.random.uniform(970.0, 1004.0, samples_per_class) # Severe drop (>2 hPa rapid drop threshold)
    c2_hrv = np.random.uniform(35.0, 65.0, samples_per_class)
    c2_hr = np.random.uniform(70.0, 95.0, samples_per_class)
    c2_spo2 = np.random.uniform(96.0, 100.0, samples_per_class)
    c2_resp = np.random.uniform(14.0, 20.0, samples_per_class)
    c2 = np.column_stack([c2_temp, c2_hum, c2_baro, c2_hrv, c2_hr, c2_spo2, c2_resp])
    y2 = np.ones(samples_per_class, dtype=np.int32) * 2

    # Class 3: Hazardous PM2.5 Pollution Spike
    c3_temp = np.random.uniform(20.0, 35.0, samples_per_class)
    c3_hum = np.random.uniform(40.0, 75.0, samples_per_class)
    c3_baro = np.random.uniform(1008.0, 1018.0, samples_per_class)
    c3_hrv = np.random.uniform(15.0, 35.0, samples_per_class)
    c3_hr = np.random.uniform(85.0, 115.0, samples_per_class)
    c3_spo2 = np.random.uniform(87.0, 93.0, samples_per_class)  # Hypoxia / SpO2 drop
    c3_resp = np.random.uniform(22.0, 32.0, samples_per_class) # Rapid shallow breathing
    c3 = np.column_stack([c3_temp, c3_hum, c3_baro, c3_hrv, c3_hr, c3_spo2, c3_resp])
    y3 = np.ones(samples_per_class, dtype=np.int32) * 3

    X_raw = np.vstack([c0, c1, c2, c3])
    y = np.concatenate([y0, y1, y2, y3])
    
    return normalize_features(X_raw), y

# 4. BUILD & TRAIN LIGHTWEIGHT KERAS MODEL
print("Generating synthetic multi-hazard telemetry dataset...")
X, y = generate_synthetic_dataset(samples_per_class=1500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training TinyML classifier model...")
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 5. QUANTIZE & EXPORT TO TFLITE (.tflite)
print("\nQuantizing model to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic range quantization
tflite_model = converter.convert()

tflite_filename = "hazard_model.tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)

file_size_kb = os.path.getsize(tflite_filename) / 1024.0
print(f"Model exported successfully: '{tflite_filename}'")
print(f"TFLite Model Size: {file_size_kb:.2f} KB (Well under 5 MB limit!)")

# 6. OFFLINE INFERENCE & EARLY WARNING PUSH NOTIFICATION SIMULATOR
def simulate_edge_inference(raw_sensor_input):
    """Simulates real-time processing on-device using TFLite Interpreter."""
    # Initialize interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess & normalize raw live input
    norm_input = normalize_features(np.array(raw_sensor_input, dtype=np.float32).reshape(1, -1))

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], norm_input)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100.0

    # Dispatch Rule-Grounded Early Warnings
    warnings = {
        0: "Status Normal: Environmental and physiological telemetry within safe bounds.",
        1: "Early Warning: Extreme Heat Stress Risk detected. High probability of heat exhaustion—hydrate immediately and move to shade.",
        2: "Early Warning: Flash Flood / Heavy Rain anomaly detected in your micro-zone. Move to higher ground.",
        3: "Early Warning: Hazardous PM2.5 Pollution Spike. Respiratory stress detected—wear an N95 mask or head indoors."
    }

    print("\n--- LIVE SENSOR STREAM TEST ---")
    print(f"Raw Input Telemetry: {raw_sensor_input}")
    print(f"Predicted Hazard Class: {predicted_class} (Confidence: {confidence:.2f}%)")
    print(f"Pushed Early Warning: \"{warnings[predicted_class]}\"")

# Test Simulation: Extreme Heatwave Input Sample
test_heatwave_telemetry = [42.5, 75.0, 1008.0, 18.0, 110.0, 96.0, 22.0]
simulate_edge_inference(test_heatwave_telemetry)