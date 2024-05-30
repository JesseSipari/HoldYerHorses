
# AI Card Bot
Final project for the Building AI course

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Summary](#Summary)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Interactive Training](#interactive-training)
  - [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

## Summary

This project aims to develop a Texas Hold'em bot that detects cards on the screen and learns through user feedback. The bot uses machine learning models to predict the suit and rank of the cards, and it can be trained interactively to improve its accuracy.

Here's how:

* Card Detection: Uses TensorFlow and Keras for image recognition. The model predicts both the suit and rank of the cards.
* Interactive Training: Uses a Tkinter GUI for user feedback. Users can confirm or correct predictions to improve the model.
* Evaluation: Evaluates the model using validation and test datasets.
* Data Handling: Custom data generator to handle both suit and rank labels. Class mappings are defined in a CSV file.

### Built With

* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Tkinter](https://docs.python.org/3/library/tkinter.html)

## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/JesseSipari/cardbot.git
   cd cardbot
   ```
2. Install the required libraries
   ```sh
   pip install tensorflow pandas pillow opencv-python tk
   ```
3. Download the dataset
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data).
   - Extract the dataset and place it in the `dataset` directory.

## Usage

### Training the Model

To train the model, run the `train_cards.py` script. This script will train the model on the dataset and save the trained model.

```sh
python train_cards.py
```

### Interactive Training

To perform interactive training with user feedback, run the `interactive_training.py` script. This script will display images and ask for user feedback to improve the model.

```sh
python interactive_training.py
```


### Evaluation

To evaluate the model's performance, use the evaluation function provided in the `train_cards.py` or `interactive_training.py` scripts. The evaluation function will print classification reports and confusion matrices for both suit and rank predictions.



## Acknowledgements

* [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data)
```
