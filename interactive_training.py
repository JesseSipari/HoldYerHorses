import os
import sys
import codecs
import numpy as np
import tensorflow as tf
import pandas as pd
import random
from tkinter import Tk, Label, Button, messagebox, OptionMenu, StringVar
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Change the default encoding to 'utf-8'
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# Load pre-trained card recognition model or create a new one
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

def load_image(path, target_size=(224, 224)):
    image = load_img(path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0

def update_model(images, suits, ranks):
    y_suits = to_categorical([SUITS.index(suit) for suit in suits], num_classes=5)
    y_ranks = to_categorical([RANKS.index(rank) for rank in ranks], num_classes=14)
    x_train = np.vstack(images)

    # Update the model with the batch of examples
    if x_train.size > 0:
        print("Training model with new data...")
        print(f"Training suits: {suits}")
        print(f"Training ranks: {ranks}")
        logs = model.fit(x_train, {'suit_output': y_suits, 'rank_output': y_ranks}, epochs=1, verbose=1)
        if logs is not None:
            model.save('card_recognition_model.keras')  # Save the updated model
        print("Model updated with new batch")

def generate_classification_report(y_true, y_pred, labels):
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    print(report)

def evaluate_model(validation_dir):
    val_images = []
    val_suits = []
    val_ranks = []

    for class_name in os.listdir(validation_dir):
        class_dir = os.path.join(validation_dir, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    image = load_image(img_path)
                    val_images.append(image)
                    suit, rank = get_suit_and_rank(class_name)
                    val_suits.append(SUITS.index(suit))
                    val_ranks.append(RANKS.index(rank))

    x_val = np.vstack(val_images)
    y_suits = np.array(val_suits)
    y_ranks = np.array(val_ranks)

    predictions = model.predict(x_val)
    pred_suits = np.argmax(predictions[0], axis=1)
    pred_ranks = np.argmax(predictions[1], axis=1)

    print("\nSuit Classification Report:")
    generate_classification_report(y_suits, pred_suits, SUITS)

    print("\nRank Classification Report:")
    generate_classification_report(y_ranks, pred_ranks, RANKS)

    # Confusion matrix for suits
    cm_suits = confusion_matrix(y_suits, pred_suits)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_suits, annot=True, fmt='d', xticklabels=SUITS, yticklabels=SUITS)
    plt.title('Suit Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Confusion matrix for ranks
    cm_ranks = confusion_matrix(y_ranks, pred_ranks)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_ranks, annot=True, fmt='d', xticklabels=RANKS, yticklabels=RANKS)
    plt.title('Rank Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Card Recognition Training")

        self.label = Label(root, text="")
        self.label.pack()

        self.image_label = Label(root)
        self.image_label.pack()

        self.feedback_label = Label(root, text="Is the prediction correct?")
        self.feedback_label.pack()

        self.yes_button = Button(root, text="Yes", command=self.yes_feedback)
        self.yes_button.pack(side='left')

        self.no_button = Button(root, text="No", command=self.no_feedback)
        self.no_button.pack(side='left')

        self.stop_button = Button(root, text="Stop", command=self.stop_feedback)
        self.stop_button.pack(side='right')

        self.back_button = Button(root, text="Back", command=self.back_feedback)
        self.back_button.pack(side='right')

        self.dataset_path = os.path.normpath('poker-card-image-recognition/dataset/training')
        self.class_names = list(CARD_TO_CLASS.keys())
        self.image_files = self.collect_image_files()
        self.current_image_file = None
        self.current_image = None
        self.predicted_label = None
        self.previous_image_file = None  # To keep track of the previous image
        self.images_to_update = []
        self.suits_to_update = []
        self.ranks_to_update = []

        self.load_next_image()

    def collect_image_files(self):
        image_files = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.dataset_path, class_name)
            if os.path.exists(class_dir):
                image_files.extend([os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        random.shuffle(image_files)
        print(f"Collected {len(image_files)} image files.")
        return image_files

    def load_next_image(self, image_file=None):
        if not image_file:
            if not self.image_files:
                messagebox.showinfo("Info", "No more images to process.")
                self.update_model_batch()
                self.root.quit()
                return
            image_file = self.image_files.pop()

        self.previous_image_file = self.current_image_file
        self.current_image_file = image_file
        self.current_image = load_image(self.current_image_file)
        predictions = model.predict(self.current_image)
        predicted_suit = SUITS[np.argmax(predictions[0])]
        predicted_rank = RANKS[np.argmax(predictions[1])]

        # Ensure correct handling of Joker
        if predicted_suit == 'joker':
            predicted_label = 'joker'
        else:
            predicted_label = f'{predicted_rank} of {predicted_suit}'

        self.predicted_label = predicted_label

        print(f'Loaded image: {self.current_image_file}, Prediction: {self.predicted_label}')  # Debugging information

        img = Image.open(self.current_image_file)
        img = img.resize((224, 224), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img
        self.label.config(text=f'Prediction: {self.predicted_label}')

    def yes_feedback(self):
        print(f"Confirmed: {self.predicted_label}")
        self.load_next_image()
    
    def no_feedback(self):
        correct_label = os.path.basename(os.path.dirname(self.current_image_file)).replace('_', ' ')
        correct_suit, correct_rank = get_suit_and_rank(correct_label)
        if correct_suit and correct_rank:
            self.images_to_update.append(self.current_image)
            self.suits_to_update.append(correct_suit)
            self.ranks_to_update.append(correct_rank)
            print(f"Corrected to: {correct_label}")
            self.load_next_image()
        else:
            messagebox.showwarning("Warning", "Invalid card name. Please try again.")

    def stop_feedback(self):
        self.update_model_batch()
        self.root.quit()
        self.root.destroy()  # Ensure the Tkinter instance is properly destroyed


    def back_feedback(self):
        if self.previous_image_file:
            self.load_next_image(self.previous_image_file)
            self.previous_image_file = None  # Clear the previous image after going back
            print("Went back to previous image.")
        else:
            messagebox.showinfo("Info", "No previous image to go back to.")

    def update_model_batch(self):
        if self.images_to_update and self.suits_to_update and self.ranks_to_update:
            print(f'Updating model with {len(self.images_to_update)} images')  # Debugging information
            update_model(self.images_to_update, self.suits_to_update, self.ranks_to_update)
            self.images_to_update.clear()
            self.suits_to_update.clear()
            self.ranks_to_update.clear()

def main():
    root = Tk()
    app = App(root)
    root.mainloop()

    # Evaluate the model during the training process
    validation_dir = 'poker-card-image-recognition/dataset/validation'
    evaluate_model(validation_dir)

    # Evaluate the model after the training process
    test_dir = 'poker-card-image-recognition/dataset/test'  # Update this path to your test dataset
    evaluate_model(test_dir)

if __name__ == "__main__":
    main()

