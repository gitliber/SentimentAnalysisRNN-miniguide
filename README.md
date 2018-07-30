# SentimentAnalysisRNN-miniguide
A step by step mini-guide about how your first RNN and use it for Sentiment Analysis

# Intro

Let me walk you through all of the steps needed to make a well working sentiment detection with Keras and long short-term memory networks. Keras is a very popular python deep learning library, similar to TFlearn that allows to create neural networks without writing too much boiler plate code. LSTM networks are a special form or network architecture especially useful for text tasks which I am going to explain later. 

# What is Keras?

Keras is an open source python library that enables you to easily build Neural Networks. The library is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, Theano, and MXNet. Tensorflow and Theano are the most used numerical platforms in Python to build Deep Learning algorithms but they can be quite complex and difficult to use. In comparison, Keras provides an easy and convenient way to build deep learning models. It’s creator François Chollet developed it to enable people to build Neural Networks as fast and easy as possible. He laid his focus on extensibility, modularity, minimalism and the support of python. Keras can be used with GPUs and CPUs and it supports both Python 2 and 3. Google Keras made a big contribution to the commoditization of deep learning and artificial intelligence since it has commoditized powerful, modern Deep Learning algorithms that previously were not only inaccessible but also unusable as well.
What is Sentiment Analysis?

# What is Sentiment Analysis

Sentiment analysis is a natural language processing problem where text is understood and the underlying intent is predicted.

With Sentiment Analysis, we want to determine the attitude (e.g the sentiment) of for example a speaker or writer with respect to a document, interaction, or event. Therefore it is a natural language processing problem where text needs to be understood, to predict the underlying intent. The sentiment is mostly categorized into positive, negative and neutral categories. With the use of Sentiment Analysis, we want to predict for example a customers opinion and attitude about a product based on a review he wrote about it. Because of that, Sentiment Analysis is widely applied to things like reviews, surveys, documents and much more.

# The imdb Dataset

This dataset contains the movie reviews from IMDb along with their associated binary sentiment polarity labels.

The imdb sentiment classification dataset consists of 50,000 movie reviews from imdb users that are labeled as either positive (1) or negative (0). The reviews are preprocessed and each one is encoded as a sequence of word indexes in the form of integers. The words within the reviews are indexed by their overall frequency within the dataset. For example, the integer “2” encodes the second most frequent word in the data. The 50,000 reviews are split into 25,000 for training and 25,000 for testing. 

The dataset was created by researchers of the Stanford University and published in a paper in 2011, where they achieved 88.89% accuracy. It was also used within the “Bag of Words Meets Bags of Popcorn” Kaggle competition in 2011.

You can use any method to classify the test data.

Below is the original dataset we used for this competition. 

http://ai.stanford.edu/~amaas/data/sentiment/

# Importing the Dependencies and getting the Data

Keras provides access to the IMDB dataset built-in.

The keras.datasets.imdb.load_data() allows you to load the dataset in a format that is ready for use in neural network and deep learning models.

The words have been replaced by integers that indicate the absolute popularity of the word in the dataset. The sentences in each review are therefore comprised of a sequence of integers.

Calling imdb.load_data() the first time will download the IMDB dataset to your computer and store it in your home directory under ~/.keras/datasets/imdb.pkl as a 32 megabyte file.

Usefully, the imdb.load_data() provides additional arguments including the number of top words to load (where words with a lower integer are marked as zero in the returned data), the number of top words to skip (to avoid the “the”‘s) and the maximum length of reviews to support.

Let’s load the dataset and calculate some properties of it. We will start off by loading some libraries and loading the entire IMDB dataset as a training dataset.

We start by importing the required dependencies to preprocess our data and to build our model.

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

We continue with downloading the imdb dataset, which is fortunately already built into Keras. Since we don’t want to have a 50/50 train test split, we will immediately merge the data into data and targets after downloading, so that we can do an 80/20 split later on.

from keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

# Exploring the Data

Now we can start exploring the dataset:

print("Categories:", np.unique(targets))
print("Number of unique words:", len(np.unique(np.hstack(data))))

Categories: [0 1]
Number of unique words: 9998

length = [len(i) for i in data]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))

Average Review length: 234.75892
Standard Deviation: 173.0

You can see in the output above that the dataset is labeled into two categories, either 0 or 1, which represents the sentiment of the review. The whole dataset contains 9998 unique words and the average review length is 234 words, with a standard deviation of 173 words.

Now we will look at a single training example:

print("Label:", targets[0])

Label: 1

print(data[0])

[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]

Above you see the first review of the dataset which is labeled as positive (1). The code below retrieves the dictionary mapping word indices back into the original words so that we can read them. It replaces every unknown word with a “#”. It does this by using the get_word_index() function.

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()]) 
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]] )
print(decoded) 

this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert # is an amazing actor and now the same being director # father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for # and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also # to the two little boy's that played the # of norman and paul they were just brilliant children are often left out of the # list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all

# Preparing the data

Now it is time to prepare our data. We will vectorize every review and fill it with zeros so that it contains exactly 10,000 numbers. That means we fill every review that is shorter than 10,000 with zeros. We do this because the biggest review is nearly that long and every input for our neural network needs to have the same size. We also transform the targets into floats.

def vectorize(sequences, dimension = 10000):
results = np.zeros((len(sequences), dimension))
for i, sequence in enumerate(sequences):
results[i, sequence] = 1
return results
 
data = vectorize(data)
targets = np.array(targets).astype("float32")

Now we split our data into a training and a testing set. The training set will contain 40,000 reviews and the testing set 10,000.

test_x = data[:10000]
test_y = targets[:10000]

train_x = data[10000:]
train_y = targets[10000:]

# Building and Training the Model

We can now build our simple Neural Network. We start by defining the type of model we want to build. There are two types of models available in Keras: the Sequential model and the Model class used with functional API.

Then we simply add the input-, hidden- and output-layers. Between them, we are using dropout to prevent overfitting. Note that you should always use a dropout rate between 20% and 50%. 

At every layer, we use “Dense” which means that the units are fully connected. Within the hidden-layers, we use the relu function, because this is always a good start and yields a satisfactory result most of the time. Feel free to experiment with other activation functions. And at the output-layer, we use the sigmoid function, which maps the values between 0 and 1. Note that we set the input-shape to 10,000 at the input-layer, because our reviews are 10,000 integers long. The input-layer takes 10,000 as input and outputs it with a shape of 50.

Lastly, we let Keras print a summary of the model we have just built.

# Input - Layer
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))

# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu")
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))

# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))model.summary()

model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 50)                500050    
_________________________________________________________________
dropout_1 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                2550      
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                2550      
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 51        
=================================================================
Total params: 505,201
Trainable params: 505,201
Non-trainable params: 0
_________________________________________________________________

Now we need to compile our model, which is nothing but configuring the model for training. We use the efficient ADAM optimization procedure. The optimizer is the algorithm that changes the weights and biases during training. We also choose binary-crossentropy as loss (because we deal with binary classification) and accuracy as our evaluation metric.

model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

We are now able to train our model. We do this with a batch_size of 32 and only for two epochs because the model can overfits if we train it longer. 

The Batch size defines the number of samples that will be propagated through the network and an epoch is an iteration over the entire training data. In general a larger batch-size results in faster training, but don’t always converges fast. A smaller batch-size is slower in training but it can converge faster. 

Again, there is a lot of opportunity for further optimization, such as the use of deeper and/or larger convolutional layers. One interesting idea is to set the max pooling layer to use an input length of 500. This would compress each feature map to a single 32 length vector and may boost performance

This is definitely problem dependent and you need to try out a few different values. If you start with a problem for the first time, I would you recommend to you to first use a batch-size of 32, which is the standard size.

results = model.fit(
 train_x, train_y,
 epochs= 2,
 batch_size = 32,
 validation_data = (test_x, test_y)
)

Train on 40000 samples, validate on 10000 samples
Epoch 1/2
40000/40000 [==============================] - 16s 403us/step - loss: 0.3195 - acc: 0.8646 - val_loss: 0.2626 - val_acc: 0.8929
Epoch 2/2
40000/40000 [==============================] - 14s 347us/step - loss: 0.2004 - acc: 0.9215 - val_loss: 0.2760 - val_acc: 0.8925
 

It is time to save and evaluate our model:

# Save the model to the models folder

Running this example fits the model and summarizes the estimated performance. We can see that this very simple model achieves a score of nearly 86.27% which is in the neighborhood of the original paper, with very little effort.

model.save("models/trained_model.h5")
print(" ")
print("Model created, trained and save on the following folder: models/trained_model.h5 with " + str(output) +" % of accuracy.")
print(" ")
print(" ")

Model created, trained and save on the following folder: models/trained_model.h5 with 89.27 % of accuracy

I’m sure we can do better if we trained this network, perhaps using a larger embedding and adding more hidden layers. Let’s try a different network type. Feel free to experiment with the hyperparameters and the number of layers.

Summary

In this Post you learned what Sentiment Analysis is and why Keras is one of the most used Deep Learning libraries. On top of that you learned that Keras made a big contribution to the commoditization of deep learning and artificial intelligence. You learned how to build a simple Neural Network with six layers that can predict the sentiment of movie reviews, which achieves a 89% accuracy. You can now use this model to also do binary sentiment analysis on other sources of text but you need to change them all to a length of 10,000 or you change the input-shape of the input layer. You can also apply this model to other related machine learning problems with only a few changes.

# Sources and resources:

https://keras.io/datasets/

https://en.wikipedia.org/wiki/Sentiment_analysis

https://machinelearningmastery.com/introduction-python-deep-learning-library-keras/

https://towardsdatascience.com/how-to-build-a-neural-network-with-keras-e8faa33d0ae4

https://www.kaggle.com/c/word2vec-nlp-tutorial/data

https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

