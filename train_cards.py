import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing import image_dataset_from_directory
import json
# Set default encoding to UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Define the model architecture
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),  # Ensure the output is flattened before feeding into Dense layers
    Dense(128, activation='relu'),
    Dense(53, activation='softmax')  # Assuming 53 classes (one for each card)
])

# Use SparseCategoricalCrossentropy for sparse labels
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dataset path
dataset_path = os.path.join('poker-card-image-recognition', 'dataset')  # Update this path
training_path = os.path.join(dataset_path, 'training')
validation_path = os.path.join(dataset_path, 'validation')
test_path = os.path.join(dataset_path, 'test')  # Add test path

# Check if paths exist
if not os.path.exists(training_path) or not os.path.exists(validation_path) or not os.path.exists(test_path):
    raise FileNotFoundError(f"The specified dataset paths do not exist: {training_path}, {validation_path}, {test_path}")

# Prepare the data with tf.data.Dataset
train_dataset = image_dataset_from_directory(
    training_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=8  # Reduce batch size to fit into limited VRAM
)

validation_dataset = image_dataset_from_directory(
    validation_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=8  # Reduce batch size to fit into limited VRAM
)

test_dataset = image_dataset_from_directory(
    test_path,
    image_size=(224, 224),
    batch_size=8  # Reduce batch size to fit into limited VRAM
)

# Optimize the datasets with prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Save the class indices
class_indices = train_dataset.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)


# Train the model
model.fit(train_dataset, validation_data=validation_dataset, epochs=5)

# Save the model
model.save('card_recognition_model.keras')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy:.4f}')





# # Increase batch size for desktop with more VRAM
# train_generator = datagen.flow_from_directory(
#     training_path,
#     target_size=(224, 224),
#     batch_size=32,  # Increase batch size
#     class_mode='categorical',
#     subset='training'
# )

# validation_generator = datagen.flow_from_directory(
#     validation_path,
#     target_size=(224, 224),
#     batch_size=32,  # Increase batch size
#     class_mode='categorical',
#     subset='validation'
# )

# # Train the model with increased workers and multiprocessing
# model.fit(train_generator, validation_data=validation_generator, epochs=10, workers=8, use_multiprocessing=True)
