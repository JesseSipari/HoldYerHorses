import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import codecs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Change the default encoding to 'utf-8'
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# Define the model architecture
def create_model():
    input_img = Input(shape=(224, 224, 3))

    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)  # Add dropout
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)  # Add dropout
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Increase dropout

    output_suit = Dense(5, activation='softmax', name='suit_output')(x)  # 4 suits + 1 joker
    output_rank = Dense(14, activation='softmax', name='rank_output')(x)  # 13 ranks + 1 joker

    model = Model(inputs=input_img, outputs=[output_suit, output_rank])
    model.compile(optimizer='adam', 
                  loss={'suit_output': 'categorical_crossentropy', 'rank_output': 'categorical_crossentropy'}, 
                  metrics={'suit_output': 'accuracy', 'rank_output': 'accuracy'})
    return model

# Load pre-trained card recognition model or create a new one
try:
    model = tf.keras.models.load_model('card_recognition_model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = create_model()

# Read the CSV file and create the mapping
cards_df = pd.read_csv('cards.csv')
CLASS_TO_CARD = {row['class index']: row['labels'] for index, row in cards_df.drop_duplicates('class index').iterrows()}
CARD_TO_CLASS = {v: k for k, v in CLASS_TO_CARD.items()}

SUITS = ['clubs', 'diamonds', 'hearts', 'spades', 'joker']
RANKS = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace', 'joker']

def get_suit_and_rank(label):
    if 'joker' in label:
        return 'joker', 'joker'
    for suit in SUITS[:-1]:  # Exclude 'joker' from this loop
        if suit in label:
            for rank in RANKS[:-1]:  # Exclude 'joker' from this loop
                if rank in label:
                    return suit, rank
    return None, None

# Custom data generator
def custom_data_generator(generator):
    while True:
        x_batch, y_batch = next(generator)
        y_suits = []
        y_ranks = []
        for y in y_batch:
            label = CLASS_TO_CARD[np.argmax(y)]
            suit, rank = get_suit_and_rank(label)
            y_suits.append(SUITS.index(suit))
            y_ranks.append(RANKS.index(rank))
        y_suits = to_categorical(y_suits, num_classes=5)
        y_ranks = to_categorical(y_ranks, num_classes=14)
        yield x_batch, {'suit_output': y_suits, 'rank_output': y_ranks}

# Prepare the data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    'poker-card-image-recognition/dataset/training',  # Update with your dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    'poker-card-image-recognition/dataset/training',  # Update with your dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Use custom data generator
train_generator = custom_data_generator(train_generator)
validation_generator = custom_data_generator(validation_generator)

# Training the model
model.fit(train_generator, validation_data=validation_generator, epochs=5, steps_per_epoch=200, validation_steps=50)
model.save('card_recognition_model.keras')  # Save the trained model

def evaluate_model(validation_dir):
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    loss, accuracy = model.evaluate(validation_generator, steps=50)
    print(f"Loss: {loss}, Accuracy: {accuracy}")


    validation_generator = custom_data_generator(validation_generator)
    loss, accuracy = model.evaluate(validation_generator, steps=50)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

# Evaluate the model
validation_dir = 'poker-card-image-recognition/dataset/validation'
evaluate_model(validation_dir)
