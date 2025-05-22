import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import os

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
interpreter.allocate_tensors()

# Get model input details (for resizing images)
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape'][1:3]  # Expected height/width

# Class names (change if yours are different)
CLASSES = ["bacterial", "fungal", "healthy"]

def take_photo():
    """Take a photo with the Pi camera."""
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    camera.release()
    if not ret:
        raise Exception("Failed to take photo")
    return frame

def predict(image):
    """Run disease prediction."""
    # Resize image to what the model expects
    img = cv2.resize(image, (input_shape[1], input_shape[0]))
    img = img.astype(np.float32) / 255.0  # Normalize (if needed)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    predicted_class = CLASSES[np.argmax(output)]
    confidence = float(np.max(output))
    return predicted_class, confidence

def main():
    print("Taking photo...")
    image = take_photo()

    print("Predicting disease...")
    disease, confidence = predict(image)

    # Save the image
    os.makedirs("captures", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"captures/{timestamp}.jpg"
    cv2.imwrite(image_path, image)

    # Print result
    print(f"Result: {disease} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    main()