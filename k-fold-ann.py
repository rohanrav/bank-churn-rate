import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values #seperating independent variables index 0-2 into array
Y = dataset.iloc[:, 13].values #seperating dependent variables into 1 dimensional array

#before we split the data, we need to deal with catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#encode catagroical data for gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#one hot encoding for countries
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
             
#must delete one of the countries one-hot encoded coloumn to avoid falling into the dummy variable trap
X = X[:, 1:]
                
#splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler #feautre scaling 
sc_x = StandardScaler() #defining scaler object
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test) 

#Defining Classifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential() 

#add input layer, and first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(p = 0.1))

#seccond hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dropout(p = 0.1))

#output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#compile the ann, applying schocastic gradient descent
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

#Make predeitions and evaluate predictions
y_pred = classifier.predict(X_test)
#since the confusion matrix only accepts binary or true or false predictions not probabilities 
#we need to convert the probabilities to true or false
y_pred = (y_pred > 0.5)

#Initialize confusion matrix - correct and incorrect predictions the classifier has made
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

print("Model Accuracy: ")
print((cm[0,0] + cm[1,1])/(2000))

#predicting churn rate for a new customer
import numpy as np
new_cust = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
new_cust = sc_x.transform(new_cust)
new_pred = classifier.predict(new_cust)
new_pred = (new_pred > 0.5) 

if new_pred:
    print("Customer will leave the bank")
else:
    print("Customer will not leave the bank")

#K-fold cross validation 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()

    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X=X_train, y=Y_train, cv = 10, n_jobs = -1)


acc_mean = accuracies.mean()
varr_mean = accuracies.std()

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()

    #add input layer, and first hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(p = 0.1))

    #seccond hidden layer
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dropout(p = 0.1))

    #output layer
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    
    #compile the ann, applying schocastic gradient descent
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop']}

#create grid search object
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, Y_train)

best_parameters = grid_search.best_params_
best_paccuracy = grid_search.best_score_
