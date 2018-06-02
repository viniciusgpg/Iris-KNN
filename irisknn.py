import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_data = pd.DataFrame(iris_data, columns=iris.feature_names)
iris_data['class'] = iris.target
iris_data.head()
#dimensions of dataset
print(iris_data.shape)
iris_data.describe()
#dividing iris_data for training and testing
from sklearn.model_selection import train_test_split
X = iris_data.values[:,0:4]
Y = iris_data.values[:,4]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state = 24)
model = KNeighborsClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
#printing the confusion matrix of y_test and k-NN predictions and your accuracy score
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

