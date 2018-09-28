#Author
#Lakshmi Praveena Rangavajhula 
#Varun Suresh Parashar - vxp171830


from sklearn import tree
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import graphviz
import sys

train_file_path = sys.argv[1]
test_file_path = sys.argv[2]
#Please chafnge the file path as required.
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

#Dropping the null values
train_data = train_data.dropna()
test_data = test_data.dropna()

train_data.rename(columns={'V6': 'v6', 'V5': 'v5','V4': 'v4', 'V3': 'v3','V2': 'v2', 'V1': 'v1','V7': 'v7'}, inplace=True)
test_data.rename(columns={'V6': 'v6', 'V5': 'v5','V4': 'v4', 'V3': 'v3','V2': 'v2', 'V1': 'v1','V7': 'v7'}, inplace=True)

#Replace the String with numeric values
train_data['v1'].replace('low',0, inplace=True)
train_data['v1'].replace('med',1, inplace=True)
train_data['v1'].replace('high',2, inplace=True)
train_data['v1'].replace('vhigh',3, inplace=True)
train_data['v2'].replace('low',0, inplace=True)
train_data['v2'].replace('med',1, inplace=True)
train_data['v2'].replace('high',2, inplace=True)
train_data['v2'].replace('vhigh',3, inplace=True)
train_data['v3'].replace('5more',5, inplace=True)
train_data['v4'].replace('more',5, inplace=True)
train_data['v5'].replace('small',0, inplace=True)
train_data['v5'].replace('med',1, inplace=True)
train_data['v5'].replace('big',2, inplace=True)
train_data['v6'].replace('low',0, inplace=True)
train_data['v6'].replace('med',1, inplace=True)
train_data['v6'].replace('high',2, inplace=True)
train_data['v7'].replace('unacc',0, inplace=True)
train_data['v7'].replace('acc',1, inplace=True)
train_data['v7'].replace('good',2, inplace=True)
train_data['v7'].replace('vgood',3, inplace=True)

#Rename the Dataset
train_dataset=train_data

#Replace the String with numeric values
test_data['v1'].replace('low',0, inplace=True)
test_data['v1'].replace('med',1, inplace=True)
test_data['v1'].replace('high',2, inplace=True)
test_data['v1'].replace('vhigh',3, inplace=True)
test_data['v2'].replace('low',0, inplace=True)
test_data['v2'].replace('med',1, inplace=True)
test_data['v2'].replace('high',2, inplace=True)
test_data['v2'].replace('vhigh',3, inplace=True)
test_data['v3'].replace('5more',5, inplace=True)
test_data['v4'].replace('more',5, inplace=True)
test_data['v5'].replace('small',0, inplace=True)
test_data['v5'].replace('med',1, inplace=True)
test_data['v5'].replace('big',2, inplace=True)
test_data['v6'].replace('low',0, inplace=True)
test_data['v6'].replace('med',1, inplace=True)
test_data['v6'].replace('high',2, inplace=True)
test_data['v7'].replace('unacc',0, inplace=True)
test_data['v7'].replace('acc',1, inplace=True)
test_data['v7'].replace('good',2, inplace=True)
test_data['v7'].replace('vgood',3, inplace=True)

test_dataset = test_data

# Splitting the input columns
X_train = train_dataset.drop('v7',axis=1)
y_train = train_dataset['v7']
X_test= test_dataset.drop('v7',axis=1)
y_test= test_dataset['v7']


#To run with the same file uncomment below
#X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.25,random_state=100)
clf = tree.DecisionTreeClassifier(criterion = "gini")
clf = clf.fit(X_train, y_train)



# To generate the decision tree
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=X_train.columns,  
                         class_names=['unacc','acc','good','vgood'],  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph

# Predicting the output for test dataset
y_pred = clf.predict(X_test)

#Printing the result.
print("Accuracy ")
print(accuracy_score(y_test,y_pred)*100)
print(" Confusion matrix ")
print( confusion_matrix(y_test, y_pred))




