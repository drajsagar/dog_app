# Dog Breed Classification
This repository is my capstone project submission for the Udacity's Data Science Nanodegree program. In this project I have built a model toidentify two different class humans and dogs by reading the user provided input. If the provided input is not what the model expecting thenthe model prints a error message


## Table of Contents
* Installation
* Project Overview
* File Descriptions
* Results
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

    haarcascades
        haarcascade_frontalface_alt.xml: a pre-trained face detector provided by OpenCV
    bottleneck_features
        DogVGG19Data.npz: pre-computed the bottleneck features for VGG-19 using dog image data including training, validation, and test
    saved_models
        VGG19_model.json: model architecture saved in a json file
        weights.best.VGG19.hdf5: saved model weights with best validation loss
    dog_app.ipynb: a notebook used to build and train the dog breeds classification model
    extract_bottleneck_features.py: functions to compute bottleneck features given a tensor converted from an image
    images: a few images to test the model manually

Note: The dog image dataset used by this project can be downloaded here: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip The human image dataset can be downloaded here: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
