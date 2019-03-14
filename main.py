# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#data processing


import pandas as pd

#pd.set_option('display.max_rows', 100)
#pd.set_option('display.max_columns', 100)
#pd.set_option('display.max_colwidth', 20)
#pd.set_option('display.width', None)


#import the data set

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values #seperating independent variables index 0-2 into array
Y = dataset.iloc[:, 13].values #seperating dependent variables into 1 dimensional array

#before we split the data, we need to deal with catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() #creating object for first catagorical data, countries
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#encode catagroical data for gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#one hot encoding for countries, see machine learning basiscs for further explaination
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
             
#must delete one of the countries one-hot encoded coloumn to avoid falling into the dummy variable trap
X = X[:, 1:]
                
#splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0) #taking train data from independent var array and dependent var array, and setting the test size to 25%
#random state parameter is not required, to just provide similar results

from sklearn.preprocessing import StandardScaler #feautre scaling 
sc_x = StandardScaler() #defining scaler object
X_train = sc_x.fit_transform(X_train) #setting the training data to fit the data and transfrom the values
X_test = sc_x.transform(X_test) #same thing as step above, not re-fitting the data because it is already fit to the data

#PART 2 - Create ANN

# - import libraries



import keras


from keras.models import Sequential
from keras.layers import Dense

#sepential module will intilalize our ANN and the dense wil help us add layers to our network


#initialize the ANN

classifier = Sequential() #craeting object

#add input layer, and first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

#seccond hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
#KERAS 2 API UPDATE --> Dense(activation="sigmoid", units=1, kernel_initializer="uniform

#compile the ann, applying schocastic gradient descent
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#fitting ANN to traning set

classifier.fit(X_train, Y_train, batch_size=10, epochs=100)








#PART 3 - MAKE predeitions and evaluate predictions

#predict test set result
y_pred = classifier.predict(X_test)
#since the confusion matrix only accepts binary or true or false predictions not probabilities 
#we need to convert the probabilities to true or false
y_pred = (y_pred > 0.5) #code is saying if the probability is bigger than 0.5 set it to true, else set it to false


#making confusion matrix to tell us how many correct and incorrect prediction the classifier has made
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

print("Model Accuracy: ")
print((1549+133)/(2000))

