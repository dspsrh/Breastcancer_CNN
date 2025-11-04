import pandas as pd
import numpy as np
import os
from glob import glob
import random
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import warnings
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings('ignore')
imagePatches = glob(r"C:\Users\dongs\Desktop\BreastCancer/IDC_regular_ps50_idx5/**/*.png", recursive=True)


class0 = [] # 0 = no cancer
class1 = [] # 1 = cancer
for filename in imagePatches:
    if filename.endswith("class0.png"):
         class0.append(filename)
    else:
        class1.append(filename)


sampled_class0 = random.sample(class0, 78786)

sampled_class1 = random.sample(class1, 78786)


from matplotlib.image import imread
import cv2

def get_image_arrays(data, label):
    img_arrays = []
    for i in data:
        if i.endswith('.png'):
            img = cv2.imread(i ,cv2.IMREAD_COLOR)
            img_sized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
            img_arrays.append([img_sized, label])
    return img_arrays


class0_array = get_image_arrays(sampled_class0, 0)
class1_array = get_image_arrays(sampled_class1, 1)

#print(class0_array.size)

combined_data = np.concatenate((class0_array, class1_array))

np.random.seed(42)
np.random.shuffle(combined_data)

print (combined_data.shape)

X = []
y = []



count=0


for features,label in combined_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, 50, 50, 3)

print(count)


count=0
ccount=0


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
for label in y_train:
    if label ==0:
        count+=1
    else:
        ccount+=1
print(count)
print(ccount)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y = to_categorical(y)


import pickle


pickle_out = open("XTrain.pickle","wb")
pickle.dump(X_train,pickle_out)
pickle_out.close()

pickle_out = open("XTest.pickle","wb")
pickle.dump(X_test,pickle_out)
pickle_out.close()

pickle_out = open("yTrain.pickle","wb")
pickle.dump(y_train,pickle_out)
pickle_out.close()

pickle_out = open("yTest.pickle","wb")
pickle.dump(y_test,pickle_out)
pickle_out.close()