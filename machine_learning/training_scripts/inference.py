import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Load your trained model
model = tf.keras.models.load_model('/machine_learning/model/lettuce_disease_model.h5')  # or best_model.h5

# Class names (must match your training labels)
class_names = ['Bacterial', 'fungal', 'healthy']  # adjust if different

# Preprocess function for individual images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Test on a single image
def test_single_image(img_path):
    # Preprocess and predict
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # Display results
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()
    
    return predicted_class, confidence

# Test on all images in test folder
def test_all_images(test_dir='/machine_learning/data/lettuce/test'):
    results = []
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        for img_name in os.listdir(class_dir)[:3]:  # Test first 3 images per class
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                print(f"\nTesting: {img_path}")
                pred, conf = test_single_image(img_path)
                results.append({
                    'image': img_path,
                    'true_class': class_name,
                    'predicted_class': pred,
                    'confidence': conf
                })
    return results

# Run inference (choose one option)

# Option 1: Test specific image
# test_single_image('/kaggle/input/lettuce-plant-disease-dataset/test/bacterial/IMG_123.jpg')

# Option 2: Test multiple images (first 3 from each class)
test_results = test_all_images()

# Print summary
print("\nTest Summary:")
for result in test_results:
    print(f"Image: {os.path.basename(result['image'])}")
    print(f"True: {result['true_class']}, Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print("-----")
