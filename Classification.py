#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:42:00 2021

@author: mohamad
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from numpy import argmax
from numpy import asarray
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QLineEdit, QMessageBox, QFileDialog
class Mlp_clsiffication:
    def __init__(self, *args, **kwargs):
        pass
    def read_file(self):
        dir_path=QFileDialog.getExistingDirectory(None,"Choose Directory","E:\\",QFileDialog.ShowDirsOnly)
        print(dir_path)
        model = load_model("model.h5")
    
        for filename in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path,filename),cv2.IMREAD_GRAYSCALE)
            plt.figure()
            
            t = np.array(img)
            t = t.astype(float) / 255.
            t=cv2.resize(t,(28,28))
            t = t.reshape(1, t.shape[0] * t.shape[1])
            yhat = model.predict(asarray(t))
            plt.imshow(img)
            if argmax(yhat)==0:
                plt.title("A")
            if argmax(yhat)==1:
                plt.title("B")
            if argmax(yhat)==2:
                plt.title("D")
            if argmax(yhat)==3:
                plt.title("G")
        plt.show()
    def Test(self):
        img = cv2.imread("test.png",cv2.IMREAD_GRAYSCALE)
        model = load_model("model.h5")
        img = np.array(img)
        img = img.astype(float) / 255.
        img=cv2.resize(img,(28,28))
        img = img.reshape(1, img.shape[0] * img.shape[1])
        
        yhat = model.predict(asarray(img))
        return argmax(yhat)
    def Train_classification(self):
        m=Mlp_clsiffication()
        y_train ,X_train=m.extrac()       
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_train = X_train.astype('float32')
# encode strings to integer
        y_train = LabelEncoder().fit_transform(y_train)
        n_features = X_train.shape[1]
        
        model = Sequential()
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
        model.add(Dense(350, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(256, activation='softmax'))
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # fit the model
        history=model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)
        loss, acc = model.evaluate(X_train, y_train, verbose=0)
        print('Test Accuracy: %.3f' % acc)
        model.save('model.h5')
        return history.history['loss'] ,loss ,acc
    
    def extrac(self):
        images=[]
        label=[]
        for filename in os.listdir("Dataset/A"):
            img = cv2.imread(os.path.join("Dataset/A",filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img,(28,28)))
                label.append(1)
        for filename in os.listdir("Dataset/B"):
            img = cv2.imread(os.path.join("Dataset/B",filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img,(28,28)))
                label.append(2)
        for filename in os.listdir("Dataset/D"):
            img = cv2.imread(os.path.join("Dataset/D",filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img,(28,28)))
                label.append(3)
        for filename in os.listdir("Dataset/j"):
            img = cv2.imread(os.path.join("Dataset/j",filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img,(28,28)))
                label.append(4)
        images = np.array(images)
        images = images.astype(float) / 255.
        
        return label ,images
        
