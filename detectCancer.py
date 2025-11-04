from sre_parse import CATEGORIES
from tokenize import Double
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from unidecode import Cache
import pickle
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


X_train = pickle.load(open("XTrain.pickle","rb"))
X_test = pickle.load(open("XTest.pickle","rb"))
y_train = pickle.load(open("yTrain.pickle","rb"))
y_test = pickle.load(open("yTest.pickle","rb"))
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("Y.pickle","rb"))
XtestD = pickle.load(open("XtestD.pickle","rb"))
ytestD = pickle.load(open("YtestD.pickle","rb"))


CATEGORIES = ["Negative","Positive"]

def prepare(filepath):
    IMG_SIZE  = 50
    img_array=cv2.imread(filepath,cv2.IMREAD_COLOR)
    new_array  =cv2.resize(img_array, (50, 50), interpolation=cv2.INTER_LINEAR)
    return new_array.reshape(-1, 50, 50, 3)

model = tf.keras.models.load_model("CancerNet.model")

prediction = model.predict([prepare('9022_idx5_x2151_y1151_class1.png')])

#print(prediction)





Y_pred = model.predict(X_train)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_train,axis = 1) 
print(X_train.shape)
print(y_train.shape)

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(Y_true,Y_pred_classes,labels=[0,1])
print('Classification report : \n',matrix)
#model.evaluate(X_test,y_test)


#if(prediction[0][0]<prediction[0][1]):
#    print("Postive")
#else:
#    print("Negative")
