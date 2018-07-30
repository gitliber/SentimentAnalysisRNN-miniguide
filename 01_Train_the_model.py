import tkinter
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb


# Import Dependencies
# matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence

import random

#Model Variables
embedding_vector_length = 32 
top_words = 5000
max_review_length = 5000


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)

data = np.concatenate((training_data, testing_data), axis=0)

targets = np.concatenate((training_targets, testing_targets), axis=0)

def vectorize(sequences, dimension = 5000):
 results = np.zeros((len(sequences), dimension))
 for i, sequence in enumerate(sequences):
  results[i, sequence] = 1
 return results
 
data = vectorize(data)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]



# Build the model

### Input - Layer
model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length)) 
### Hidden - Layers
model.add(LSTM(100)) 
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))


### Output- Layer
model.add(Dense(1, activation='sigmoid'))
print(model.summary()) 


# compiling the model
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)
results = model.fit(
 train_x, train_y,
 epochs= 2,
 batch_size = 64,
 validation_data = (test_x, test_y)
)

# Spliting data for test and training

from keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)


#Print Accuracy of the model

from decimal import Decimal, ROUND_HALF_UP
our_value = Decimal(np.mean(results.history["val_acc"])*100)
output = Decimal(our_value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))

# Save the model to the models folder
model.save("models/trained_model.h5")
print(" ")
print("Model created, trained and save on the following folder: models/trained_model.h5 with " + str(output) +" % of accuracy.")
print(" ")
print(" ")
