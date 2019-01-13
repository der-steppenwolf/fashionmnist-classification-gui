# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:41:44 2018

@author: Theodora Panou
"""

import tensorflow as tf 
from tensorflow import keras
import time

# Global variables
learning_rate = 0.01
epochs = 10

class FashionClassification:
    
    """Provide a Neural Network trained with fashion MNIST and evaluated on accuracy, training time and mean prediction time."""
    
    # class variable with the 10 classes/labels names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def buildNN(self):
        """
        .. function:: buildNN(epochs)
            Return a built, trained and compiled neural network with 3 hidden 
            layers of 128 nodes each.
            
            :return: a compiled and trained neural network
            :rtype: model built with the keras API
        """
        
        (X_train, y_train, X_test, y_test) = self._loadFashionMNIST()
        nn = self._model()
        self._train(nn, X_train, y_train, X_test, y_test)
        return nn
    
    def _loadFashionMNIST(self):
        """
        .. function:: _loadFashionMNIST()
        
            Return fashion MNIST scaled train and test data with their 
            respective labels, in four ndarrays.
            
            :returns: the training images, training labels, 
                      test images and test labels
            :rtype: ndarray, [n_images, n_rows, n_columns] 
                    ndarray, [n-images, n_labels] 
                    ndarray, [n_images, n_rows, n_columns] 
                    ndarray, [n-images, n_labels] 
        """
        
        # Images are 28x28 NumPy arrays, with pixel values from 0 to 255. 
        # The labels are an array of integers from 0 to 9
        f_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = f_mnist.load_data() 
        
        # Scale px values to 0 and 1 before feeding to Neural Network
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        
        return (train_images, train_labels, test_images, test_labels)
            
    def _model(self):
        """
        .. function:: _model()
            
                Build a Neural Network, configure the model's layers and return 
                the compiled the model.
                
                :returns: a compiled Neural Network
                :rtype: model built with the keras API
        """
        
        model = keras.Sequential([
            # Input layer : Flatten 2D to 1D array of 28 x 28 pixels
            keras.layers.Flatten(input_shape=(28, 28)), # input shape required
            # Hidden layer I: ReLu 128-node layer
            keras.layers.Dense(128, activation=tf.nn.relu),
            # Hidden layer II: ReLu 128-node layer
            keras.layers.Dense(128, activation=tf.nn.relu),
            # Hidden layer III: ReLu 128-node layer
            keras.layers.Dense(128, activation=tf.nn.relu),
            # Output layer : 10-node softmax layer
            # return 10 probabilities corresponding to the 10 classes
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
            
        # Add optimizer, loss function and metrics
        model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate), 
                  loss='sparse_categorical_crossentropy',
                  # accuracy: the fraction of correctly classified images 
                  metrics=['accuracy'])
        
        return model
    
    def _train(self, model, X_train, y_train, X_test, y_test):
        """
        .. function:: _train(model, X_train, y_train, X_test, y_test, epochs)
            
            Train model with the training images and labels. Evaluate accuracy 
            for 1000 test images and labels and display training time and mean 
            prediction time of the images in seconds.
            
            :param model: the built neural network
            :param X_train: the training images
            :param y_train: the training labels
            :param X_test: the test images
            :param y_test: the test labels
            :type model: 
            :type X_train: ndarray, [n_images, n_pixels]
            :type y_train: ndarray, [n_images]
            :type X_test: ndarray, [n_images, n_pixels]
            :type y_test: ndarray, [n_images]
        """
        
        start = time.time()
        # Train model with train examples
        model.fit(X_train, y_train, epochs=epochs, shuffle=True)
        end = time.time()
        # Display NN training time
        print('\nNN Training Time:', end - start, end=' sec\n\n')
