# Flower Image Classifier with Deep Learning - TensorFlow  


## Project Background

Project to develop a deep neural network with TensorFlow that recognizes the type of a flower using 102 different types of flowers, where there are ~20 images per flower to train on.

There are two parts of this project:

- Part1 aims to build and train a custom image classifier using transfer learning technique with MobileNet architecture as a pre-trained model and Dense layer with softmax function activation for the predictions. The new neural network was trained on Oxford Flowers 102 dataset and then saved with the .h5 extension in this repository.

- Part2 aims to convert the developed and saved Keras model from part1 into a command-line application.


## Files

- ` Project_Image_Classifier_Project.ipynb ` : Main jupyter notebook demonstrates all the steps to create and train a TensorFlow Keras model with MobileNet to predict the type of a flower from its image. 

- ` Project_Image_Classifier_Project.html ` : HTML version of the jupyter notebook. 

- ` predict.py ` : Python command line application that uses the trained deep learning network from the jupyter notebook and classifies an input flower image.

- ` saved_keras_model.h5 ` : The trained and saved deep learning model that predicts the type of a flower.

- ` label_map.json ` : Mapping labels to the actual flower names.


## Technologe library

- Python
- Numpy
- PIL
- Matplotlib
- Tensonflow
- Tensorflow Hub
- Anaconda (Jypyter Notebook)
- Argparse
- GPU required


### Data

This project uses Oxford Flowers 102 dataset and it was divided into a training set, a validation set, and a test set.

The Oxford Flowers dataset is so large I cannot upload it onto this repository but you can find the data (here)[https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html], thus, in order to run this project, you will need to use a GPU device. 


## Useg

### Part-1

To run ` Project_Image_Classifier_Project.ipynb ` you will need to have software installed to run and execute an iPython Notebook

I recommend Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

To start a notebook server using a command-line interface, open the Terminal or Anaconda prompt(for Windows), navigate to the directory where all the project's files reside, and type:


``` jupyter notebook Project_Image_Classifier_Project.ipynb ```


The above command will start and open the Jupyter Notebook server in the current directory in a new tab in your browser.


### Part-2 

The ` predict.py ` module will read in an image and the saved Keras model and then predicts the top flower names from an image along with their corresponding probabilities.

There are 4 images provided in the ` ./test_images/ ` folder to check the predict.py script. The 4 images are:
- cautleya_spicata.jpg
- hard-leaved_pocket_orchid.jpg
- orange_dahlia.jpg
- wild_pansy.jpg


To run the predict.py script successfully, type on the terminal or command line one of the following commands: 

- Basic usage: 

you have to pass as an arguments 1- the path to an image and 2- the path to the saved Keras model

``` python predict.py "./test_images/orange_dahlia.jpg" "./saved_keras_model.h5" ```


- Options usage:

` --top_k ` : This argument returns the top K most likely classes, and ~~ --category_names ~~ : The path to a JSON file mapping labels to flower names

``` python predict.py "./test_images/orange_dahlia.jpg" "./saved_keras_model.h5" --top_k 3 --category_names "./label_map.json" ``` 

