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

def load_image(image_path):
    """Load an existing image file instead of using camera"""
    if not os.path.exists(image_path):
        raise Exception(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Failed to load image: {image_path}")
    return image

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
    # Specify the path to your test image
    test_image_path = "test_image.jpg"  # Change this to your image file name
    
    print(f"Loading test image: {test_image_path}")
    try:
        image = load_image(test_image_path)
        
        print("Predicting disease...")
        disease, confidence = predict(image)

        # Save a copy of the processed image (optional)
        os.makedirs("captures", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"captures/processed_{timestamp}.jpg"
        cv2.imwrite(result_path, image)

        # Print result
        print(f"Result: {disease} (Confidence: {confidence:.2%})")
        print(f"Processed image saved to: {result_path}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()