# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 17:29:47 2019

@author: Theodora Panou
"""

import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
from PIL import Image, ImageTk
from classification import FashionClassification

# class variable with the 10 classes/labels names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def trainNN():
    """Train neural network and set global variable to the trained model."""
    
    # build neural network and train it in the background
    imc = FashionClassification()
    global nn
    nn = imc.buildNN()

def browseFiles():
    """Open filedialogue, display and classify selected image on button click."""
    
    fname = askopenfilename(filetypes=(("JPG files", "*.jpg"),
                                       ("PNG files", "*.png"),
                                       ("JPEG files", "*.jpeg")))
    if fname:
        try:
            img = Image.open(fname)
        except:
            showerror("Open Source File", "Failed to read file\n'%s'" % fname)
        
        # Get resized grayscale of image to feed to NN
        data = preprocess(img)
        
        # Resize oringinal image
        basewidth = 200
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        
        # Get tk image of resized original
        photo = ImageTk.PhotoImage(img)
        
        # Display tk image in label
        imgLabel.configure(image=photo)
        imgLabel.image = photo
        
        # Predict label for image
        predictions = nn.predict(data)
        for _, logits in enumerate(predictions):
            name = class_names[np.argmax(logits)]
            botLabel['text'] = "uploaded image is a {}".format(name)

def preprocess(img):
    """Preprocess image before feeding it to neural network."""
    
    X = []
    # Get gray-scale image in mode F
    img_gray = img.convert('F')
    
    # Resize grayscale to 28 x 28
    img_gray = img_gray.resize((28, 28), Image.ANTIALIAS)
    
    # PIL Image to ndarray, [M, N], returns img with black background
    img_gray = np.array(img_gray)
    
    # Change black background to white with a defined object outline.
    for i in range(28):
        for j in range(28):
            if img_gray[i,j] > 200.0:
                img_gray[i,j] = 0.0
    
    # ndarray, [M, N] to 2D list
    img_gray = img_gray.tolist()
    
    X.append(img_gray)
    
    # Return ndarray, [n_files, M, N]    
    return np.array(X)

# Create tk App Window
root = tk.Tk()   
root.title("Fashion MNIST Classification - gui")

# Create top, mid and bottom frames
topFrame = tk.Frame(root)
topFrame.pack(side=tk.TOP)

midRFrame = tk.Frame(root, height=50, width=300)
midRFrame.pack_propagate(0) # don't shrink
midRFrame.pack(side=tk.RIGHT)

botFrame = tk.Frame(root)
botFrame.pack(side=tk.BOTTOM)

# Create labels for the frames
topLabel1 = tk.Label(topFrame, text="Image Classification Demo using the fashion MNIST dataset", font="Times 15")

topLabel2 = tk.Label(topFrame, text=("\n1. Upload an image (JPG, JPEG, PNG) of clothing belonging" 
                                      "\n          to any one of the following 10 categories: T-shirt/top, Trouser," 
                                      "\n           Pullover, Dress, Coat, Sandal, Shirt,Sneaker, Bag, Ankle boot"
                                      "\n\n  2. Get output label from trained and compiled Neural Network"),
                            font="Times 12",
                            padx=20)
topLabel1.pack()
topLabel2.pack()

# Bottom box for predictions
botLabel = tk.Label(botFrame, text="Label Y", bg="white", fg="black",
                    font="Times 13", borderwidth=2, relief="groove", padx=80)
botLabel.pack()

# Label image placeholder
sprite = ImageTk.PhotoImage(Image.open('placeholder.png'))
imgLabel = tk.Label(image=sprite)
imgLabel.image = sprite
imgLabel.pack(side=tk.LEFT, padx=60, pady=10)

# Button to upload image
btn = tk.Button(midRFrame, text="select image file", command=browseFiles, 
                fg="white", bg="black", font="Roboto 11 bold", relief="solid")
btn.pack(fill=tk.BOTH, expand=True, padx=80, pady=5)

# Min window size
root.minsize(300, 500)

# Train NN after 0.5 sec
root.after(500, trainNN)

root.mainloop()