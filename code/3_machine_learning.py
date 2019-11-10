#%% import module

# basic module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklaern module
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#%% load data
file_path = 'C:/Users/ssmoo/Desktop/kaggle_titanic_prediction/data/refined_data/'
train_X = pd.read_csv(file_path + 'train_X.csv')
train_y = pd.read_csv(file_path + 'train_y.csv')
test_X = pd.read_csv(file_path + 'test_X.csv')

#%% train test split
X_train, X_test, y_train, y_test = train_test_split(train_X.values, train_y.values,
                                                    random_state = 0)

#%% knn
n_neighbors_array = np.arange(1, 101)
n_neighbors_score_array = []

for n_neighbors in n_neighbors_array:
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(X_train, y_train)
    n_neighbors_score_array.append(knn.score(X_test, y_test))

np.array(n_neighbors_score_array)

plt.plot(n_neighbors_array, n_neighbors_score_array)

#%% naive_bayes

nb = GaussianNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)

#%% logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

#%% SVM
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

#%% Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt.score(X_test, y_test)
