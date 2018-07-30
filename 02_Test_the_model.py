# Import Dependencies
import tkinter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
from data import encode_sentence
from keras.models import load_model
from keras.utils import to_categorical
from keras import models
from keras import layers

# Spliting data for test and training

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)


## Verifiy the model

from tkinter import Tk, Label, Button, Message
class MyFirstGUI:
    def __init__(self, master):
        master.geometry("600x600")
        self.master = master
        master.title("Neural Network Lesson by Jair Ribeiro")

        self.label = Label(master, text="Sentiment Analysis with Keras")
##        self.label = Label(master, text="Current Model Accuracy: " + str(output) +" %")        
        self.label.pack()


        # Checking the data 
        ReviewNumber= random.randint(1, 1000)
        index = imdb.get_word_index()
        reverse_index = dict([(value, key) for (key, value) in index.items()]) 
        decoded = " ".join( [reverse_index.get(i - ReviewNumber, "#") for i in data[0]] )

        self.label = Label(master, text="Analyzing randomly the Review n. " + str(ReviewNumber) + " from IMDB")
        self.label.pack()

        self.label = Label(master, text="Unknown words were replaced with #")
        self.label.pack()
        
        self.message = Message(master, text=" " + str(decoded))
        self.message.pack()


        # Exploring the data
        self.label = Label(master, text="Exploring the data")
        self.label.pack()
        self.label = Label(master, text="Categories: " + str(np.unique(targets)))
        self.label.pack()       

        Sentiment = "Positive" if targets[0] > 0 else "Negative"
        self.label = Label(master, text="Sentiment Analysis: " + Sentiment + " (" + str(targets[0])+ ").")
        self.label.pack()
 
        self.label = Label(master, text="Analyzing the data.")
        self.label.pack()    

        self.label = Label(master, text="Number of unique words: " + str(len(np.unique(np.hstack(data))))+".")
        self.label.pack()
        
        self.label = Label(master, text="Averages")
        self.label.pack()

        length = [len(i) for i in data]
        self.label = Label(master, text="Average Review length: " + str("%6.0f" % np.mean(length))+" words.")
        self.label.pack()
        
        self.label = Label(master, text="Standard Deviation: " + str("%6.0f" % round(np.std(length)))+" words.")
        self.label.pack()


        
### Close Button
        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()



root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()


