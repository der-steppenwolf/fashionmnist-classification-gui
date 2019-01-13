# Image Classification with Fashion MNIST

Basic gui that classifies images of clothing and displays the predicted labels y, using a neural network created with tensorflow and trained with [fashion MNIST](https://github.com/zalandoresearch/fashion-mnist). The project was created for training purposes for an ML university course, as well as to learn python 3.6

## Table of contents

* Technologies
* Getting Started
* Authors
* License

## Technologies

* python 3.6

## Geting Started

To get the project running you need python 3.6 and the packages outlined in the `requirements.txt` file installed. The installation guide bellow requires [Anaconda](https://www.anaconda.com/download/). 
If you do not have Anaconda installed feel free to run the `gui.py` script assuming you've installed all required packages.

### Installation

##### Clone

* Clone this repo to your local machine using ```https://github.com/der-steppenwolf/fashionmnist-classification-gui.git```

##### Anaconda

This will create a new virtual environment (venv) and install all required python packages to it using the `requirements.txt` file. If you have an active venv with the required packages installed skip to changing directory into the cloned repository and running `gui.py` with python.

* In the anaconda prompt create a new environment

```
conda create -n <venv name> python=3.6
```

* Activate the new environment

```
activate <venv name>
```

* cd to the cloned repository

```
cd path\to\repository
```

* Add all specs in `requirements.txt` to the new venv. The necessary packages will be installed or updated.

```
conda install --file requirements.txt
```

* Run the `gui.py` script located in the cloned repository to start using the gui.

```
python gui.py
```

Finally, a minimal gui window created with the tkinter library should appear and the anaconda prompt should show the following verbose training message during which the window won't be responsive:


```
Epoch 1/10
2019-01-13 16:49:28.806143: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
60000/60000 [==============================] - 5s 86us/step - loss: 0.7468 - acc: 0.7481
Epoch 2/10
60000/60000 [==============================] - 5s 81us/step - loss: 0.4802 - acc: 0.8313
Epoch 3/10
60000/60000 [==============================] - 5s 87us/step - loss: 0.4341 - acc: 0.8469
Epoch 4/10
60000/60000 [==============================] - 5s 80us/step - loss: 0.4053 - acc: 0.8572
Epoch 5/10
60000/60000 [==============================] - 5s 77us/step - loss: 0.3854 - acc: 0.8628
Epoch 6/10
60000/60000 [==============================] - 5s 76us/step - loss: 0.3688 - acc: 0.8686
Epoch 7/10
60000/60000 [==============================] - 5s 77us/step - loss: 0.3547 - acc: 0.8731
Epoch 8/10
60000/60000 [==============================] - 5s 78us/step - loss: 0.3436 - acc: 0.8772
Epoch 9/10
60000/60000 [==============================] - 5s 77us/step - loss: 0.3322 - acc: 0.8808
Epoch 10/10
60000/60000 [==============================] - 5s 77us/step - loss: 0.3235 - acc: 0.8833

NN Training Time: 47.921220541000366 sec
```

After the neural network is trained for 10 epochs the gui can be used to open image files (JPEG, JPG, PNG) and classify them to the 10 available categories/labels in the fashion MNIST dataset.

## Authors

* Theodora Panou

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
