from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


# Step 1: Load the existing dataset
existing_dataset = pd.read_csv('thermal_images.csv')

# Step 2: Load the new dataset
new_dataset = pd.read_csv('thermal_images.csv')

# Step 3: Merge the new dataset with the existing dataset
updated_dataset = pd.concat([plant_disease, thermal_images], axis=0, ignore_index=True)

# Step 4: Save the updated dataset if necessary
updated_dataset.to_csv('updated_PlantDisease.csv', index=False)


# Step 5: Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Step 6: Train the recommendation model
model = KNNBasic()
model.fit(trainset)



df = pd.read_csv('/Users/dheerajchetti/Downloads/updated_PlantDisease.csv')
df.head()
df.tail()
df.size
df.shape
df.columns
df['label'].unique()
df.dtypes
df['label'].value_counts()
#sns.heatmap(df.corr(),annot=True)
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
# Initialzing empty lists to append all model's name and corresponding name
acc = []
model = []
# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))
from sklearn.model_selection import cross_val_score
# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)
score
import pickle
# Dump the trained Naive Bayes classifier with Pickle
DT_pkl_filename = 'C:/Users/chandana/Desktop/Harvestify/models/DecisionTree.pkl'
# Open the file to save as pkl file
DT_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(DecisionTree, DT_Model_pkl)
# Close the pickle instances
DT_Model_pkl.close()
from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))
# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,features,target,cv=5)
score

