# Dog Breed Classification using CNN
This repository is my capstone project submission for the Udacity's Data Science Nanodegree program. In this project I have built a model toidentify two different class humans and dogs by reading the user provided input. If the provided input is not what the model expecting thenthe model prints a error message


## Table of Contents
* Installation
* Project Overview
* File Descriptions
* Results
* Blog post (https://medium.com/@drajsagar/dog-human-error-image-classification-using-cnn-8e85fe6b1782)
* Licensing, Authors, and Acknowledgements

## Installation
Beyond the Anaconda distribution of Python, the following packages need to be installed:

*    opencv-python==3.2.0.6
*    h5py==2.6.0
*    matplotlib==2.0.0
*    numpy==1.12.0
*    scipy==0.18.1
*    tqdm==4.11.2
*    scikit-learn==0.18.1
*    keras==2.0.2
*    tensorflow==1.0.0 `

## Project Overview
In this project, I built and trained a neural network model with CNN (Convolutional Neural Networks) transfer learning, using 8351 dog images of 133 breeds. CNN is a type of deep neural networks, which is commonly used to analyze image data. Typically, a CNN architecture consists of convolutional layers, activation function, pooling layers, fully connected layers and normalization layers. Transfer learning is a technique that allows a model developed for a task to be reused as the starting point for another task. If an image of a human is supplied, the code will identify the most resembling dog breed.

## File Descriptions 
Below are main foleders/files for this project:

### 1. haarcascades
* haarcascade_frontalface_alt.xml: a pre-trained face detector provided by OpenCV
### 2. bottleneck_features
* DogResnet50Data.npz: pre-computed the bottleneck features for Resnet50 using dog image data including training, validation, and test
### 3. saved_models
* weights.best.Resnet50.hdf5: saved model weights with best validation loss
### 4. dog_app.ipynb: a notebook used to build and train the dog breeds classification model
### 5. extract_bottleneck_features.py: functions to compute bottleneck features given a tensor converted from an image
### 6. images: a few images to test the model manually

Note: The dog image dataset used by this project can be downloaded here: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip The human image dataset can be downloaded here: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip


## Results

1. The model was able to reach an accuracy of 77.8708% on test data.
2. If a dog image is supplied, the model gives a prediction of the dog breed.
3.The model is also able to identify the most resembling dog breed of a person.

## Licensing, Authors, and Acknowledgements
Credits must be given to Udacity for the starter codes and data images used by this project.
