#%% import module

# basic module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklaern module
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
n_neighbors_list = np.arange(1, 101)
n_neighbors_train_score_list = []
n_neighbors_test_score_list = []

for n_neighbors in n_neighbors_list:
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(X_train, y_train)
    n_neighbors_train_score_list.append(knn.score(X_train, y_train))
    n_neighbors_test_score_list.append(knn.score(X_test, y_test))

diff_train_test_list = np.array(n_neighbors_train_score_list) - np.array(n_neighbors_test_score_list)
np.argmin(diff_train_test_list)

plt.plot(n_neighbors_list, n_neighbors_train_score_list, label = 'Train score')
plt.plot(n_neighbors_list, n_neighbors_test_score_list, label = 'Test score')
plt.axvline(x = 36, color = 'red', ls = '--')
plt.title('Train score & Test score')
plt.xlabel('n_neighbors')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()

# n_neighbors = 36
knn = KNeighborsClassifier(n_neighbors = 36)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

#%% naive_bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)

#%% logistic Regression
C_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
C_train_score_list = []
C_test_score_list = []

for C in C_list:
    lr = LogisticRegression(C = C)
    lr.fit(X_train, y_train)
    C_train_score_list.append(lr.score(X_train, y_train))
    C_test_score_list.append(lr.score(X_test, y_test)) # 0.1 에서 가장 높음

C_list_str = list(map(str, C_list))

plt.plot(C_list_str, C_train_score_list, label = 'Train score')
plt.plot(C_list_str, C_test_score_list, label = 'Test score')
plt.title('Train score & Test score')
plt.xlabel('C')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()

lr = LogisticRegression(C = 0.1)
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

#%% SVM

C_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
gamma_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
C_gamma_matrix = np.zeros([9, 9])

for j in range(len(gamma_list)):
    
    for i in range(len(C_list)):
        svm = SVC(C = C_list[i], gamma = gamma_list[j])
        svm.fit(X_train, y_train)
        C_gamma_matrix[i, j] = svm.score(X_test, y_test)

C_gamma_df = pd.DataFrame(C_gamma_matrix)
C_gamma_df.index = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
C_gamma_df.columns = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

sns.heatmap(C_gamma_df, annot=True, cmap='YlGnBu')
plt.title('Score with C, gamma')
plt.xlabel('gamma')
plt.ylabel('C')
plt.show()

# C = 1, gamma = 0.05
svm = SVC(C = 1, gamma = 0.05)
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

#%% Decision Tree
max_depth_list = np.arange(1, 101)
max_depth_train_score_list = []
max_depth_test_score_list = []

for max_depth in max_depth_list:
    dt = DecisionTreeClassifier(max_depth = max_depth)
    dt.fit(X_train, y_train)
    max_depth_train_score_list.append(dt.score(X_train, y_train))
    max_depth_test_score_list.append(dt.score(X_test, y_test))
  
max_depth_list[np.argmax(max_depth_test_score_list)]

plt.plot(max_depth_list, max_depth_train_score_list, label = 'Train score')
plt.plot(max_depth_list, max_depth_test_score_list, label = 'Test score')
plt.axvline(10, c = 'red', ls = '--')
plt.title('Train score & Test score')
plt.xlabel('Depth')
plt.ylabel('Score')
plt.grid()
plt.legend()
plt.show()

# max_depth = 10
dt = DecisionTreeClassifier(max_depth = 10)
dt.fit(X_train, y_train)
dt.score(X_test, y_test)

#%% random forest
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

#%% ensemble
def get_ensemble_pred(knn_pred, nb_pred, lr_pred, svm_pred, dt_pred, rf_pred):
    tmp_pred = knn_pred + nb_pred + lr_pred + svm_pred + dt_pred + rf_pred
    tmp_pred = (tmp_pred / 5) > 0.5
    
    return tmp_pred.astype(int)
    
knn_pred = knn.predict(X_train)
nb_pred = nb.predict(X_train)
lr_pred = lr.predict(X_train)
svm_pred = svm.predict(X_train)
dt_pred = dt.predict(X_train)
rf_pred = rf.predict(X_train)

ensemble_pred = get_ensemble_pred(knn_pred, nb_pred, lr_pred, svm_pred, dt_pred, rf_pred)

print(np.sum(ensemble_pred == y_train.flatten()) / len(X_train))

knn_pred = knn.predict(X_test)
nb_pred = nb.predict(X_test)
lr_pred = lr.predict(X_test)
svm_pred = svm.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

ensemble_pred = get_ensemble_pred(knn_pred, nb_pred, lr_pred, svm_pred, dt_pred, rf_pred)

print(np.sum(ensemble_pred == y_test.flatten()) / len(X_test))

#%% make csv file

PassengerId = np.arange(892,892 + 418)

# knn
Survived = knn.predict(test_X)
knn_pred = pd.DataFrame({'PassengerId' : PassengerId,
                         'Survived' : Survived})
knn_pred.to_csv('knn_pred.csv', index = False)

# naive bayes
Survived = nb.predict(test_X)
nb_pred = pd.DataFrame({'PassengerId' : PassengerId,
                         'Survived' : Survived})
nb_pred.to_csv('nb_pred.csv', index = False)

# logistic regression
Survived = lr.predict(test_X)
lr_pred = pd.DataFrame({'PassengerId' : PassengerId,
                         'Survived' : Survived})
lr_pred.to_csv('lr_pred.csv', index = False)

# svm
Survived = svm.predict(test_X)
svm_pred = pd.DataFrame({'PassengerId' : PassengerId,
                         'Survived' : Survived})
svm_pred.to_csv('svm_pred.csv', index = False)

# decision tree
Survived = dt.predict(test_X)
dt_pred = pd.DataFrame({'PassengerId' : PassengerId,
                         'Survived' : Survived})
dt_pred.to_csv('dt_pred.csv', index = False)

# random forest
Survived = rf.predict(test_X)
rf_pred = pd.DataFrame({'PassengerId' : PassengerId,
                         'Survived' : Survived})
rf_pred.to_csv('rf_pred.csv', index = False)

# ensemble
knn_pred = knn.predict(test_X)
nb_pred = nb.predict(test_X)
lr_pred = lr.predict(test_X)
svm_pred = svm.predict(test_X)
dt_pred = dt.predict(test_X)
rf_pred = rf.predict(test_X)

Survived = get_ensemble_pred(knn_pred, nb_pred, lr_pred, svm_pred, dt_pred, rf_pred)
ensemble_pred = pd.DataFrame({'PassengerId' : PassengerId,
                              'Survived' : Survived})
ensemble_pred.to_csv('ensemble_pred.csv', index = False)
