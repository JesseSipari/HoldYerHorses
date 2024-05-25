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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# Custom data generator function
def custom_data_generator(file_paths, labels, batch_size, img_size):
    def load_image(path):
        image = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        return image / 255.0

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def parse_function(filename, label):
        image = tf.numpy_function(load_image, [filename], tf.float32)
        image.set_shape([img_size[0], img_size[1], 3])  # Set the shape explicitly
        return image, label

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # Ensure the dataset repeats to avoid running out of data
    return dataset

# Prepare the data with data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use a portion of the data for validation
)

train_generator = datagen.flow_from_directory(
    'poker-card-image-recognition/dataset/training',  # Update with your dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    'poker-card-image-recognition/dataset/training',  # Update with your dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Extract file paths and labels
train_file_paths = train_generator.filepaths
train_labels = [CLASS_TO_CARD[class_idx] for class_idx in train_generator.classes]
validation_file_paths = validation_generator.filepaths
validation_labels = [CLASS_TO_CARD[class_idx] for class_idx in validation_generator.classes]

# Convert labels to categorical
train_labels = [get_suit_and_rank(label) for label in train_labels]
validation_labels = [get_suit_and_rank(label) for label in validation_labels]
train_suits = to_categorical([SUITS.index(suit) for suit, rank in train_labels], num_classes=5)
train_ranks = to_categorical([RANKS.index(rank) for suit, rank in train_labels], num_classes=14)
validation_suits = to_categorical([SUITS.index(suit) for suit, rank in validation_labels], num_classes=5)
validation_ranks = to_categorical([RANKS.index(rank) for suit, rank in validation_labels], num_classes=14)

# Create datasets
train_dataset = custom_data_generator(train_file_paths, {'suit_output': train_suits, 'rank_output': train_ranks}, batch_size=32, img_size=(224, 224))
validation_dataset = custom_data_generator(validation_file_paths, {'suit_output': validation_suits, 'rank_output': validation_ranks}, batch_size=32, img_size=(224, 224))

# Training the model
model.fit(train_dataset, validation_data=validation_dataset, epochs=3, steps_per_epoch=40, validation_steps=40)
model.save('card_recognition_model.keras')  # Save the trained model

def evaluate_model(validation_dir):
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Ensure class_mode=None for multi-output model
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),  # Ensure the target size matches the model's expected input shape
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    
    # Custom function to get true labels from the directory structure
    def get_labels_from_directory(directory):
        labels = []
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('png', 'jpg', 'jpeg')):
                    label = os.path.basename(subdir)
                    suit, rank = get_suit_and_rank(label)
                    if suit is not None and rank is not None:
                        labels.append((SUITS.index(suit), RANKS.index(rank)))
        return labels

    # Generate true labels
    true_labels = get_labels_from_directory(validation_dir)
    y_true_suits = tf.keras.utils.to_categorical([label[0] for label in true_labels], num_classes=5)
    y_true_ranks = tf.keras.utils.to_categorical([label[1] for label in true_labels], num_classes=14)

    # Predict using the model
    predictions = model.predict(validation_generator, steps=len(validation_generator))
    pred_suits = predictions[0]
    pred_ranks = predictions[1]

    print("\nSuit Classification Report:")
    print(classification_report(np.argmax(y_true_suits, axis=1), np.argmax(pred_suits, axis=1), target_names=SUITS))

    print("\nRank Classification Report:")
    print(classification_report(np.argmax(y_true_ranks, axis=1), np.argmax(pred_ranks, axis=1), target_names=RANKS))

    # Confusion matrix for suits
    cm_suits = confusion_matrix(np.argmax(y_true_suits, axis=1), np.argmax(pred_suits, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_suits, annot=True, fmt='d', xticklabels=SUITS, yticklabels=SUITS)
    plt.title('Suit Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Confusion matrix for ranks
    cm_ranks = confusion_matrix(np.argmax(y_true_ranks, axis=1), np.argmax(pred_ranks, axis=1))
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_ranks, annot=True, fmt='d', xticklabels=RANKS, yticklabels=RANKS)
    plt.title('Rank Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    validation_dir = 'poker-card-image-recognition/dataset/validation'
    evaluate_model(validation_dir)

