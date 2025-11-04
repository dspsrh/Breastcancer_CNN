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

print(y_train[2])