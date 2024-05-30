

```markdown
<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->


# HoldYerHorses

Final project for the Building AI course

## Summary

This project is about training a model to classify playing cards. The model is trained on a dataset of card images and can be further improved through interactive training with user feedback.

## Background

This project solves the problem of classifying playing cards based on images. This could be useful in various card games where automatic card recognition could enhance the gaming experience. My personal motivation is my interest in AI and its applications in gaming.

* Problem 1: Classifying playing cards based on images
* Problem 2: Improving the model through interactive training

## How is it used?

The model is trained using the [`train_cards.py`](train_cards.py) script. For interactive training with user feedback, the [`interactive_training.py`](interactive_training.py) script is used. The users are developers or AI enthusiasts who want to experiment with image classification and interactive training.

```sh
python train_cards.py
python interactive_training.py


## Data sources and AI methods

The data for this project comes from the [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data) on Kaggle.

The AI methods used in this project include Convolutional Neural Networks (CNNs) for image classification, and Transfer Learning to leverage pre-trained models for better performance and efficiency.

## Challenges

This project does not solve the problem of recognizing cards in real-world conditions, such as different lighting conditions or angles. Ethical considerations include ensuring that the technology is not used for cheating in card games.

## What next?

The project could be expanded to recognize other types of cards or to recognize cards in real-world conditions. Another next step is to adapt the model to detect cards from a specific game. Skills in image processing and deep learning would be useful for this.

## Acknowledgments

* [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data)
```

