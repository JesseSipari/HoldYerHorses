import os
import sys
import cv2
import numpy as np
import pyautogui
import tensorflow as tf
import matplotlib.pyplot as plt

# Set default encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Screen capture function
def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Load pre-trained card recognition model
try:
    model = tf.keras.models.load_model('model_1.h5')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error loading model: {e}")

# Function to process the image and detect cards
def detect_cards(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 20:  # Ignore very small detections
            continue
        card = frame[y:y+h, x:x+w]
        card = cv2.resize(card, (224, 224))  # Resize to match model input size
        card = np.expand_dims(card, axis=0) / 255.0  # Normalize
        prediction = model.predict(card)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class
        cards.append((x, y, w, h, predicted_class))

        # Draw rectangle around the detected card
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return cards, frame

# Example usage
if __name__ == "__main__":
    region = (0, 0, 800, 600)  # Define the region of the screen to capture
    frame = capture_screen(region)
    cards, processed_frame = detect_cards(frame)
    for (x, y, w, h, predicted_class) in cards:
        print(f"Detected card at ({x}, {y}, {w}, {h}) with predicted class: {predicted_class}")
    
    # Display the frame with detected cards outlined using Matplotlib
    plt.imshow(processed_frame)
    plt.title('Detected Cards')
    plt.axis('off')
    plt.show()
