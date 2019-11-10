#%% import module

# basic module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# sklearn module
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso

#%% set directory
os.chdir('C:/Users/ssmoo/Desktop/kaggle_titanic_prediction')
data_path = 'C:/Users/ssmoo/Desktop/kaggle_titanic_prediction/data/'

#%% load data
raw_train = pd.read_csv(data_path + 'raw_data/train.csv')
raw_test = pd.read_csv(data_path + 'raw_data/test.csv')

#%% copy data
train = raw_train.copy()
test = raw_test.copy()

#%% info
train.info()
test.info()

#%% train data preprocessing
embarked_notnull_idx = train['Embarked'].notnull()
train = train[embarked_notnull_idx]

#%% test data preprocessing
PE = test[(test['Pclass'] == 3) & (test['Embarked'] == 'S')]
fare_median = np.median(PE[PE['Fare'].notnull()]['Fare'])
test['Fare'][152] = fare_median

#%% make preprocessing function

def preprocessing(titanic_X):
    '''
    불필요한 변수 PassengerId 변수와
    결과변수인 Survived 변수를 제외한 요인변수 데이터를 입력받아 전처리 해준다.
    '''
    # 불필요한 titanic 변수 제거 (Ticket, Cabin)
    del titanic_X['Ticket'], titanic_X['Cabin']
    
    # 결측치 제거
    titanic_X = titanic_X[titanic_X['Embarked'].notnull()]
    
    # Name 변수 -> Position 변수로 변경
    def get_position(name):
        position_compile = re.compile('[A-z]*\.')
        position = position_compile.findall(name)
        
        return position[0][:-1]
    
    titanic_X['Position'] = titanic_X['Name'].apply(lambda name : get_position(name))
    del titanic_X['Name']
    
    # Name 변수값 Mr, Mrs, Miss, Master로 축약
    Mr_list = ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']
    Miss_list = ['Countess', 'Lady', 'Mlle', 'Mme', 'Ms', 'Dona']
    
    for position in Mr_list:
        titanic_X.loc[titanic_X['Position'] == position, 'Position'] = 'Mr'
        
    for position in Miss_list:
        titanic_X.loc[titanic_X['Position'] == position, 'Position'] = 'Miss'
    
    titanic_X.loc[(titanic_X['Sex'] == 'male') & (titanic_X['Position'] == 'Dr'), 'Position'] = 'Mr'
    titanic_X.loc[(titanic_X['Sex'] == 'female') & (titanic_X['Position'] == 'Dr'), 'Position'] = 'Miss'
    
    list(set(titanic_X['Position']))
    
    # SibSp, Parch -> SibSp_existence, Parch_existence 변수로 변경
    def get_existence(num):
        if num == 0:
            return 0
        elif num >= 1:
            return 1
        else:
            return None
    
    titanic_X['SibSp_existence'] = titanic_X['SibSp'].apply(lambda num : get_existence(num))
    titanic_X['Parch_existence'] = titanic_X['Parch'].apply(lambda num : get_existence(num))
    
    del titanic_X['SibSp'], titanic_X['Parch']
    
    # Fare 변수 scaling
    robust = RobustScaler()
    titanic_X['Fare'] = robust.fit_transform(titanic_X[['Fare']])
    
    # one_hot encoding_1
    feature_list = ['Pclass', 'Embarked', 'Position', 'SibSp_existence', 'Parch_existence']
    for feature in feature_list:
        titanic_X[feature] = titanic_X[feature].astype(str)
    
    titanic_X = pd.get_dummies(titanic_X)
    
    # Age 변수 결측값 예측
    age_notnull = titanic_X[titanic_X['Age'].notnull()]
    age_isnull = titanic_X[titanic_X['Age'].isnull()]
    age_notnull_X = age_notnull.iloc[:, 1:]
    age_notnull_y = age_notnull.iloc[:, 0]
    age_isnull_X = age_isnull.iloc[:, 1:]
    lasso = Lasso(alpha = 0.01)
    lasso.fit(age_notnull_X.values, age_notnull_y.values)
    age_isnull['Age'] = lasso.predict(age_isnull_X.values)
    
    titanic_X = pd.concat([age_notnull, age_isnull]).sort_index()
    
    # Age 변수 -> Aged 변수로 변경
    def get_aged(age):
        if age < 13:
            return 0
        elif age < 20:
            return 1
        elif age < 30:
            return 2
        elif age < 45:
            return 3
        elif age < 60:
            return 4
        elif age >= 60:
            return 5
        else:
            return None
    
    titanic_X['Aged'] = titanic_X['Age'].apply(lambda age : get_aged(age))
    del titanic_X['Age']
    titanic_X['Aged'] = titanic_X['Aged'].astype(str)
    
    # one-hot encoding_2
    titanic_X = pd.get_dummies(titanic_X)
    
    return titanic_X

train_X = train.iloc[:, 2:]
train_X = preprocessing(train_X)
train_y = train.iloc[:, [1]]

test_X = test.iloc[:, 1:]
test_X = preprocessing(test_X)

#%% save data
train_X.to_csv('train_X.csv', index = False)
train_y.to_csv('train_y.csv', index = False)
test_X.to_csv('test_X.csv', index = False)