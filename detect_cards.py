import sys
import os
import codecs
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Change the default encoding to 'utf-8'
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# Load your trained model
model = tf.keras.models.load_model('card_recognition_model.keras')

def preprocess_frame(frame):
    # Resize the frame to match the model's expected input shape
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize the frame (assuming the model expects normalized input)
    normalized_frame = resized_frame / 255.0
    return normalized_frame

def detect_cards(frame):
    # This is a simplified example for detecting multiple cards.
    # In a real application, you would likely use a more complex method
    # to identify and segment each card from the frame.

    # For now, assume we are looking for up to 4 cards in fixed locations.
    card_locations = [
        (50, 50, 150, 200),  # (top-left-x, top-left-y, width, height)
        (200, 50, 150, 200),
        (350, 50, 150, 200),
        (500, 50, 150, 200)
    ]

    detected_cards = []
    for (x, y, w, h) in card_locations:
        card_image = frame[y:y+h, x:x+w]
        processed_frame = preprocess_frame(card_image)
        processed_frame = np.expand_dims(processed_frame, axis=0)

        predictions = model.predict(processed_frame)
        
        suits_prediction = predictions[0]
        ranks_prediction = predictions[1]
        
        predicted_suit = np.argmax(suits_prediction, axis=1)[0]
        predicted_rank = np.argmax(ranks_prediction, axis=1)[0]
        
        card = map_class_to_card(predicted_suit, predicted_rank)
        detected_cards.append(card)

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{card}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return detected_cards, frame

def map_class_to_card(predicted_suit, predicted_rank):
    SUITS = ['clubs', 'diamonds', 'hearts', 'spades', 'joker']
    RANKS = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace', 'joker']
    
    return f"{RANKS[predicted_rank]} of {SUITS[predicted_suit]}"
