import sys
import os
import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model

# Add YOLOv7 repo to Python path
sys.path.append(os.path.join(os.getcwd(), 'yolov7'))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

# Ensure YOLOv7 model file exists
def check_file(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist. Please download it or place it in the correct directory.")

# Load your trained model for classification
classification_model = load_model('card_recognition_model.keras')  # Update this path if necessary

# Ensure YOLOv7 model file exists
check_file(os.path.join('yolov7', 'yolov7.pt'))

# Manually load YOLOv7 model
model = attempt_load(os.path.join('yolov7', 'yolov7.pt'), map_location='cpu')  # Adjust map_location as needed
model.eval()

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    return normalized_frame

def detect_cards(frame):
    # Resize frame to YOLOv7 expected input size
    img_size = 640  # Example size, you might need to adjust this based on your YOLOv7 configuration
    resized_frame = cv2.resize(frame, (img_size, img_size))
    img = torch.from_numpy(resized_frame).to('cpu')  # Use appropriate device (e.g., 'cuda' for GPU)
    img = img.permute(2, 0, 1).float()  # Change from HWC to CHW
    img /= 255.0
    img = img.unsqueeze(0)
    
    print(f"Input image shape: {img.shape}")

    with torch.no_grad():
        results = model(img)
    
    results = results[0] if isinstance(results, tuple) else results
    print(f"YOLOv7 results shape: {results.shape}")
    print(f"YOLOv7 raw results: {results}")  # Print raw results for debugging

    detections = non_max_suppression(results, 0.25, 0.45, classes=0, agnostic=False)
    if detections is None or len(detections) == 0 or detections[0] is None:
        print("No detections")
        return [], frame

    detections = detections[0]
    print(f"Detections shape: {detections.shape}")

    detections = detections.cpu().numpy()  # Convert to numpy array

    detected_cards = []

    for detection in detections:
        if len(detection) < 6:
            print(f"Unexpected detection shape: {detection.shape}")
            continue
        
        x1, y1, x2, y2, conf, cls = detection[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(f"Detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}, cls={cls}")

        card_image = frame[y1:y2, x1:x2]
        if card_image.size == 0:
            continue

        processed_frame = preprocess_frame(card_image)
        processed_frame = np.expand_dims(processed_frame, axis=0)

        predictions = classification_model.predict(processed_frame)
        suits_prediction = predictions[0]
        ranks_prediction = predictions[1]

        # Print prediction shapes
        print(f"suits_prediction shape: {suits_prediction.shape}, ranks_prediction shape: {ranks_prediction.shape}")

        if suits_prediction.shape[1] != 5 or ranks_prediction.shape[1] != 14:
            print(f"Unexpected prediction shape: suits_prediction {suits_prediction.shape}, ranks_prediction {ranks_prediction.shape}")
            continue

        predicted_suit = np.argmax(suits_prediction, axis=1)[0]
        predicted_rank = np.argmax(ranks_prediction, axis=1)[0]

        card = map_class_to_card(predicted_suit, predicted_rank)
        detected_cards.append(card)

        # Draw bounding box and label on the frame
        label = f"{card}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return detected_cards, frame

def map_class_to_card(predicted_suit, predicted_rank):
    SUITS = ['clubs', 'diamonds', 'hearts', 'spades', 'joker']
    RANKS = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace', 'joker']
    return f"{RANKS[predicted_rank]} of {SUITS[predicted_suit]}"

if __name__ == "__main__":
    # Example usage
    cap = cv2.VideoCapture(0)  # Change to 0 or path to your video file if necessary

    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        sys.exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        print("Frame captured.")
        detected_cards, frame_with_detections = detect_cards(frame)
        print("Detections processed.")

        cv2.imshow('Frame', frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
