# What version of Python do you have?
import sys

import tensorflow.keras #import keras won't work we have to import keras as import tensorflow.keras to acess keras gpu processing library
import pandas as pd
import sklearn as sk

#imports related to classifications
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as pyplot #this module will help us to plot grid and visualize our dataset and stuff
from matplotlib import style #this is gonna change the style of our grid
import pickle #this module will help us to save our Ml model once the training is complete

print(f"Tensor Flow Version: {tf.__version__}")
print(f"tensor-flow (GPU): {tf.test.gpu_device_name()}") #check if the tensorflow gpu is installed or not
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"numpy {np.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# now that we've checked for the pakages and their version installed in the conda environment




#before we can use car.data we need to add attributes buying,maint,door,persons,lug_boot,safety,class on the first line in the file cars.data file
#so the reason we are doing this is because we need pandas to read this file and what padas does is that it reads the first line of any input file as an attribute or the features for the dataset
# so I have just defined what these attributes are by adding the line buying,maint,door,persons,lug_boot,safety,class on the first line in the file cars.data file

#now we can go ahead and read the cars.data file using pandas
#here we will be using read_csv even though it's not a csv file but the data inside this file is seperated by comma so we are going to use it
data = pd.read_csv("car.data")

#just to ensure it's working I am going to go ahead and print out the data head here
print(data.head())

#We should generally avoid using attributes with non numerical data for example yes or no in it because we are performing computaion on this data and we are performing operations on them
#and we cannot do that on a non-numerical data of an attribute in a data file
#but here we are dealing with a data file in which most of the attributes that it's containing is of non-numerical type that means there is only one way to deal with it
#we need to convert the non numerical data into a numerical data in all the attributes of the data file that have them
#so inorder to convert all thoes non numerical attribute to numeric attribute we are going to use sklearn preprocessing for that
#vhigh,vhigh,2,2,small,low,unacc so here we are going to convert vhigh,vhigh,small,low,unacc into integer values that corresponds with the medium
#so all of our med = 1 low = 0 and high = 2 and the same thing for all other attributes as well and sklearn has a preprocessing module whch will help us to do that

#its gonna take the lables in our data with non integer data in it and encode it in their appropriate integer values
#not at this time this is just the object that will do this for us we haven't done it yet we need to pass our dataframe to it to actually do that
Preprocessing_data = preprocessing.LabelEncoder()
#create an list for each of our columns in the data preprocessing requires a list
#so we are going to read the data file using pandas
buying = Preprocessing_data.fit_transform(list(data["buying"]))
maint = Preprocessing_data.fit_transform(list(data["maint"]))
door = Preprocessing_data.fit_transform(list(data["door"]))
persons = Preprocessing_data.fit_transform(list(data["persons"]))
lug_boot = Preprocessing_data.fit_transform(list(data["lug_boot"]))
safety = Preprocessing_data.fit_transform(list(data["safety"]))
class_ = Preprocessing_data.fit_transform(list(data["class_"]))

#after conversion is complete we need to add this back into our main list
#buying = Preprocessing_data.fit_transform(list(data["buying"])) is gonna return to us a numpy array
#so now that we have integers we can work with this data

#here create a predict variable (output after the training of our KNN ML model is complete
predict = "class_"

X = list(zip(buying, maint, door, persons, lug_boot, safety))#attributes we will use zip to turn all of our attributes into one list
Y = list(class_)#labels turn our class_ into a list

# now we are going to split X and Y arrays into 4 variables
# X test , Y test , X train and Y train
# here sk = sklearn
# so here essentially we are taking all of our attributes in X array and all of our labels that we are trying to predict and we are going to split them up into
# four different arrays
# x_train array is gonna be a section of X attribute array
# y_train array is gonna be a section of Y label array
# x_test, y_test arrays are our test data that we are gonna use to test the accuracy of our Ml model that we are gonna create
# now the way it works is if we trained the model every single bit of data that we have and it will simply just memorize it
# test_size = 0.1 is splitting 10% of our data into a test samples (x_test, y_test arrays) so that when we test of that and it's never seen that information before

# if you use x_train, y_train, x_test, y_test = sk.model_selection.train_test_split(X, Y, test_size = 0.4 )
# then it will throw an error Input contains NaN, infinity or a value too large for dtype('float64').
# change the above to this x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size = 0.4 ) and this will solve the issue
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size=0.1)

#now we can pass this data to the classifier for our ML model training

#create our KNN classifier
classifier_ml_model = KNeighborsClassifier(n_neighbors=5) #this will take in the amount of neighbour K's values

#training the KNN ML model
classifier_ml_model.fit(x_train, y_train)

#testing the accuracy of our trained KNN ML model
accuracy = classifier_ml_model.score(x_test, y_test)
print(f"accuracy of KNN ML model = {accuracy}%")

#Performing predictions using our model
predicted = classifier_ml_model.predict(x_test)
#final labels that will be used by the model as an output data
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print(f"Predicted data = {names[predicted[x]]} , Data = {x_test[x]}, Actual answers = {names[y_test[x]]}")