#import neccessary Libraries and modules
import pandas as pd
import numpy as np
import os
from glob import glob
import random
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, SeparableConv2D
from tensorflow.keras.callbacks import EarlyStopping


import warnings


import pickle


X_train = pickle.load(open("XTrain.pickle","rb"))
X_test = pickle.load(open("XTest.pickle","rb"))
y_train = pickle.load(open("yTrain.pickle","rb"))
y_test = pickle.load(open("yTest.pickle","rb"))

model = Sequential()


model.add(SeparableConv2D(64, (3, 3), padding="same",input_shape=(50,50,3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(SeparableConv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(SeparableConv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(SeparableConv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.20))
model.add(SeparableConv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(SeparableConv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(SeparableConv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))


model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(Dense(2))
model.add(Activation("softmax"))

epochs = 50  

opt = Adagrad(learning_rate=1e-2, decay=1e-2 / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
print(model.summary())
 
batch_size = 250
history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs= epochs,verbose = 2,batch_size=batch_size)   
#history = tf.keras.models.load_model("CancerNet.model")



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save('CancerNet.model')