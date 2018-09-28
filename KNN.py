#Author
# Lakshmi Praveena Rangavajhula - lxr171030
# Varun Suresh Parashar - vxp171830


# Packages we will use
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import sys

# Read the Input arguments
train_file_path = sys.argv[1]
test_file_path = sys.argv[2]

# Read the Test and Train file

trainData = pd.read_csv(train_file_path)
testData = pd.read_csv(test_file_path)


#Headers for the data
headers=['instant1','instant2', 'dteday', 'season', 'yr', 'mnth', 'hr',
       'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp',
       'hum', 'windspeed', 'casual', 'registered', 'cnt']

# Apply headers for the dataset
trainData.columns = headers
testData.columns = headers


#Remove redundant Columns
redundantColumns = ['instant1', 'instant2','dteday','casual', 'registered']
trainData.drop(redundantColumns, inplace=True,axis = 1)
testData.drop(redundantColumns, inplace=True,axis = 1)

#Remove the records with null values
trainData = trainData.dropna()
testData = testData.dropna()
#Split the data 
trainDataX = trainData.drop(trainData.columns[-1],axis=1)
trainDataY = trainData[trainData.columns[-1]]
testDataX = testData.drop(testData.columns[-1],axis=1)
testDataY = testData[testData.columns[-1]]

print("Linear Regression")
#  linear regression object
regr = linear_model.LinearRegression()
regr.fit(trainDataX, trainDataY)
countPrediction = regr.predict(testDataX)

# The mean squared error
print("Mean squared error without k-5 fold :"
    , mean_squared_error(testDataY, countPrediction))

# Ridge
from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.001, normalize=False)
ridgeReg.fit(trainDataX, trainDataY)
ridgepred = ridgeReg.predict(testDataX)

print("Mean squared error with Ridge:"
      , mean_squared_error(testDataY, ridgepred))


#Lasso

from sklearn.linear_model import Lasso
lassoReg = Lasso(alpha=0.000001, normalize=True)
lassoReg.fit(trainDataX, trainDataY)
lassopred = lassoReg.predict(testDataX)
print("Mean squared error with Lasso :"
      , mean_squared_error(testDataY, lassopred))

#for K5 Validation
dataX = trainDataX.append(testDataX)
dataY = trainDataY.append(testDataY)


from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

kf.get_n_splits(dataX)

MSE = 0.0
for train_index, test_index in kf.split(dataX):
    regr.fit(dataX.iloc[train_index], dataY.iloc[train_index])
    countPrediction = regr.predict(dataX.iloc[test_index])
    MSE = MSE + mean_squared_error(dataY.iloc[test_index], countPrediction)

print ("Mean square error with K-5 Fold : ", MSE/5)


# Now let's build a MLP regressor ...
from sklearn.neural_network import MLPRegressor

print("\n Neural Network")

MLPregr = MLPRegressor(hidden_layer_sizes=(30,1000,30), max_iter=150000)

MLPregr.fit(trainDataX,trainDataY)
predictions = MLPregr.predict(testDataX)
MSE = sum((predictions - testDataY)**2)/len(predictions)
print("Means square error without k-5 fold : ",MSE)

# Now we compute the models and average the MSEs:
MSE = 0.0
for train_index, test_index in kf.split(dataX):
    MLPregr.fit(dataX.iloc[train_index], dataY.iloc[train_index])
    predictions = MLPregr.predict(dataX.iloc[test_index])
    MSE += sum((predictions - dataY.iloc[test_index])**2)/len(predictions)
    
print("Means square error with k-5 fold : ",MSE/5.0)


print("\n KNN ")
from sklearn.neighbors import KNeighborsRegressor
KNNRegr = KNeighborsRegressor(n_neighbors=5)
KNNRegr.fit(trainDataX,trainDataY)


predictions = KNNRegr.predict(testDataX)
MSE = sum((predictions - testDataY)**2)/len(predictions)
print ("Means square error without k-5 fold using 5 neighbours:",MSE)


MSE = 0
for train_index, test_index in kf.split(dataX):
   KNNRegr = KNeighborsRegressor(n_neighbors=5)
   KNNRegr.fit(dataX.iloc[train_index], dataY.iloc[train_index])
   predictions = KNNRegr.predict(dataX.iloc[test_index])
   MSE += sum((predictions - dataY.iloc[test_index])**2)/len(predictions)
print("Means square error with k-5 fold using 5 neighbours:",MSE/5.0)
  
