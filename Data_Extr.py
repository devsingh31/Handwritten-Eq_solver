import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd

def load_images_from_folder(folder):
    train_data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {filename}")
            continue
        img = ~img
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        maxi = 0
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            maxi = max(w*h, maxi)
            if maxi == w*h:
                x_max = x
                y_max = y
                w_max = w
                h_max = h
        im_crop = thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
        im_resize = cv2.resize(im_crop, (28, 28))
        im_resize = np.reshape(im_resize, (784, 1))
        train_data.append(im_resize)
    return train_data

# Assign labels to data and concatenate
def assign_labels_and_concatenate(folder, label):
    data = load_images_from_folder(folder)
    for i in range(len(data)):
        data[i] = np.append(data[i], [label])
    return data

if __name__ == "__main__":
    base_path = '/Users/devsingh/Desktop/ECS174_FInal_Project_Equation/extracted_images'  # Base path where your images are stored

    # Load and label data
    data_minus = assign_labels_and_concatenate(os.path.join(base_path, '-'), '10')
    data_plus = assign_labels_and_concatenate(os.path.join(base_path, '+'), '11')
    data_times = assign_labels_and_concatenate(os.path.join(base_path, 'times'), '12')
    data_digits = []
    for digit in range(10):  # Assuming digits 0-9 are in folders named '0' through '9'
        digit_data = assign_labels_and_concatenate(os.path.join(base_path, str(digit)), str(digit))
        data_digits.append(digit_data)

    # Concatenate all data
    data = np.concatenate([data_minus, data_plus, data_times] + data_digits)

    print(f"Total number of images processed: {len(data)}")

    # Save to CSV
    df = pd.DataFrame(data, index=None)
    df.to_csv('train_final.csv', index=False) # used for training
