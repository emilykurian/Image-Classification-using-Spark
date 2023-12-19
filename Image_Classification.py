# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:08:44 2023

@author: emyes
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from matplotlib import image
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
import csv

# Set up Spark session
spark = SparkSession.builder.appName("DogsVsCats").getOrCreate()

# Import images
dir = 'C:/Users/emyes/OneDrive/Desktop/Big Data/Assignment 4/train/'
img_dir = 'C:/Users/emyes/OneDrive/Desktop/Big Data/Assignment 4/test1/'

train_img = ['C:/Users/emyes/OneDrive/Desktop/Big Data/Assignment 4/train/{}'.format(i) for i in os.listdir(dir)]


classes = []

for p in os.listdir(dir):
    category = p.split(".")[0]
    if category =='dog':
        classes.append(1)
    else:
        classes.append(0)
        
df = pd.DataFrame({
    'filename': train_img,
    'class': classes
})

print(df.shape)

df['class'].value_counts().plot.bar()

# Image Preprocessing

p =cv2.imread(train_img[0])
pp = cv2.resize(p,(32,32))
plt.imshow(pp)

def image_to_feature_vector(image, size=(32, 32)):

    return cv2.resize(image, size).flatten() 
# “feature vector” will be a list of 32 x 32 x 3 = 3,072 numbers

# histogram Equalization
columns = 3
show =15
plt.figure(figsize=(20,15))

for idx, img in enumerate(train_img):
    
    if idx >= show:
        break
    temp = cv2.imread(img)
    
    #  ## Color Images
    hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    eq_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    color = ('b','g','r')
    plt.subplot(int(np.ceil(show / columns) ), columns, idx + 1)
    for i,col in enumerate(color):
        histr = cv2.calcHist([eq_color],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
plt.show()

def hist_eq(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    eq_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return  cv2.resize(eq_color, (32,32)).flatten() 

rawImages = []
rawImages2 = []
labels = []

for i in range(len(df)):
    img=cv2.imread(train_img[i])
    pixels = image_to_feature_vector(img)
    hist = hist_eq(img)
    rawImages.append(pixels)
    rawImages2.append(hist)
    labels.append(df['class'].loc[i])

print(np.shape(rawImages))
print(np.shape(rawImages2))
print(np.shape(labels))

# Apply KNN

k = round(np.sqrt(25000))
k

(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.3, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(rawImages2, labels, test_size=0.3, random_state=42)


model = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
model.fit(trainFeat, trainLabels)
print(model.score(trainFeat, trainLabels))
acc = model.score(testFeat, testLabels)
print("histogram accuracy: {:.2f}%".format(acc * 100))


# Load test images
test_img = ['C:/Users/emyes/OneDrive/Desktop/Big Data/Assignment 4/test1/{}'.format(j) for j in os.listdir(img_dir)]

testImages = []

for i in range(len(test_img)):
     img = cv2.imread(test_img[i])
     hist = hist_eq(img)
     testImages.append(hist)

#Predict labels for test images
predictedLabels = model.predict(testImages)

#write the predicted labels to a CSV file
with open('predicted_labels2.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     writer.writerow(['ImageId', 'Label'])
     for i, label in enumerate(predictedLabels):
         writer.writerow([i+1, label])    

################################PART 2 #################################################
        
# Load oxford test images

dataset_dir = 'C:/Users/emyes/OneDrive/Desktop/Big Data/Assignment 4/oxford images/' 

oxford_list = []
img_names = []
actualLabels = []

for file_name in os.listdir(dataset_dir):
    if file_name.endswith('.jpg'): # Assuming all images in the dataset are in JPG format
        file_path = os.path.join(dataset_dir, file_name)
        img = cv2.imread(file_path)
        # Check if image was loaded correctly
        if img is not None:
            hist = hist_eq(img)
            oxford_list.append(hist)
            img_names.append(file_name)
            # Determine actual label based on first letter of file name
            if file_name[0].isupper():
                actualLabels.append('cat')
            else:
                actualLabels.append('dog')
        else:
            print(f'Failed to load image: {file_name}')
            continue # Skip to the next iteration of the loop
        

# Predict labels for test images
oxpredictedLabels = model.predict(oxford_list)

# Replace 1 and 0 with 'dog' and 'cat'
oxpredictedLabels = ['dog' if label == 1 else 'cat' for label in oxpredictedLabels]

# Calculate accuracy of predictions
accuracy = np.mean(np.array(oxpredictedLabels) == np.array(actualLabels))

# Write the predicted labels and actual labels to a CSV file
with open('oxford_predicted_labels_final.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImageName', 'PredictedLabel', 'ActualLabel'])
    for i, (predicted, actual) in enumerate(zip(oxpredictedLabels, actualLabels)):
        writer.writerow([img_names[i], predicted, actual])

print(f'Accuracy: {accuracy:.2f}')


