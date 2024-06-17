import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import matplotlib.pyplot as plt


# Load the trained model
def load_model():
    with open('model_final.json', 'r') as json_file: # saved from training
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('model_final.weights.h5')
    return model


# Preprocess the input image and visualize the segments
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    img = ~img
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    images = []
    plt.figure(figsize=(10, 2))
    for i, c in enumerate(cnt):
        x, y, w, h = cv2.boundingRect(c)
        im_crop = thresh[y:y + h, x:x + w]
        im_resize = cv2.resize(im_crop, (28, 28))
        im_resize = np.reshape(im_resize, (28, 28, 1))
        images.append(im_resize)

        # Visualize each segment
        plt.subplot(1, len(cnt), i + 1)
        plt.imshow(im_crop, cmap='gray')
        plt.axis('off')
    plt.show()

    images = np.array(images)
    return images


# Recognize and print the equation
def recognize_equation(model, images):
    classes = '0123456789+-x'
    predictions = model.predict(images)
    equation = ''
    for prediction in predictions:
        label = np.argmax(prediction)
        equation += classes[label]
    return equation


# Evaluate the equation
def evaluate_equation(equation):
    equation = equation.replace('x', '*')
    try:
        result = eval(equation)
        return result
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    model = load_model()
    image_path = '/Users/devsingh/Desktop/ECS174_FInal_Project_Equation/test_images/test8.jpg'  # Replace with the path to your image
    images = preprocess_image(image_path)
    equation = recognize_equation(model, images)
    print("Recognized Equation:", equation)
    result = evaluate_equation(equation)
    print("Evaluation Result:", result)
