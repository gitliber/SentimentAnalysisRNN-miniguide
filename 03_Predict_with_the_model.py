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
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence
from numpy import array

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
        master.title("Neural Network Lesson: Predicting Sentiments")

        self.label = Label(master, text="Sentiment Analysis with Keras")
##        self.label = Label(master, text="Current Model Accuracy: " + str(output) +" %")        
        self.label.pack()


        # Checking the data

        import keras
        NUM_WORDS=5000 # only use top 1000 words
        INDEX_FROM=3   # word index offset
        word_to_id = keras.datasets.imdb.get_word_index()
        word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        id_to_word = {value:key for key,value in word_to_id.items()}    
        review_len = 10000
        max_review_length = review_len
        model = load_model(("models/trained_model.h5"))
        
        ReviewNumber= random.randint(1, 1000)
        index = imdb.get_word_index()
        reverse_index = dict([(value, key) for (key, value) in index.items()]) 
        decoded = " ".join( [reverse_index.get(i - ReviewNumber, "#") for i in data[0]] )
 
        self.label = Label(master, text="Analyzing the data.")
        self.label.pack()    

        self.label = Label(master, text="Number of unique words: " + str(len(np.unique(np.hstack(data))))+".")
        self.label.pack()

        length = [len(i) for i in data]
        self.label = Label(master, text="Average Review length: " + str("%6.0f" % np.mean(length))+" words.")
        self.label.pack()
        
        self.label = Label(master, text="Standard Deviation: " + str("%6.0f" % round(np.std(length)))+" words.")
        self.label.pack()

# Using the model to predict sentiments on two specific phrases. 

        bad = "this movie was terrible and bad"
        good = "the movie is amazing i really liked the movie and had fun"

        for review in [bad]:
            tmp = []
        for word in review.split(" "):
            tmp.append(word_to_id[word])
            tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
        self.message = Message(master, text="Predicting the Sentiment in a specific phrase: " + str((review,model.predict(array([tmp_padded][0]))[0][0]))+".")
        self.message.pack()

        if model.predict(array([tmp_padded][0])) > 0.99999 :
         sentiment = "Negative"
 
        if model.predict(array([tmp_padded][0])) < 0.99999 :
           sentiment ="Positive"

        self.label = Label(master, text="Sentiment: " + sentiment)
        self.label.pack()            
        
### Close Button
        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()


