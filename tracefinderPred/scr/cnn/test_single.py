import numpy as np
import tensorflow as tf
import cv2
import os
from feature_extraction import extract_noise
from preprocess import load_and_preprocess

# Paths
MODEL_PATH = "processed/dual_branch_cnn.h5"
LABELS = sorted(os.listdir("data/Official"))   # assumes Official/<device_folder>

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_single(image_path):
    # Preprocess official image
    img = load_and_preprocess(image_path, size=(256,256), gray=False)
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # Extract noise map
    noise = extract_noise(image_path, size=(256,256))
    noise = np.expand_dims(noise, axis=0)  # add batch dimension

    # Predict
    preds = model.predict({"official_input": img, "noise_input": noise})
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    return LABELS[pred_class], confidence

if __name__ == "__main__":
    test_image = "data/Official/device_1/150/sample_001.png"  # example path
    device, conf = predict_single(test_image)
    print(f"Predicted Device: {device} (Confidence: {conf:.2f})")
