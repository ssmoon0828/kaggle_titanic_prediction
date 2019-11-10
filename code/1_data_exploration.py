#%% import module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family = 'Gulim')
import seaborn as sns
import os
import re

#%% set directory
os.chdir('C:/Users/ssmoo/Desktop/kaggle_titanic_prediction')
data_path = 'C:/Users/ssmoo/Desktop/kaggle_titanic_prediction/data/'

#%% load data
raw_titanic = pd.read_csv(data_path + 'raw_data/train.csv')
titanic = raw_titanic.copy()

#%% information
titanic.info()
# PassengerId : 승객번호
# Survivied : 생존여부 -> y값
# Pclass : 선실등급
# Name : 이름 -> 성, 직위를 추출할 수 있다.
# Sex : 성별
# Age : 나이 -> 그룹화?, 결측값 수 : 177
# SibSp : 배우자, 형제자매 수
# Parch : 부모, 자녀 수
# Ticket : 티켓 번호 -> 여기서 특볋나 정보를 찾기는 쉽지 않을것 같다.
# Fare : 운임 요금 -> 선실등급과 상관관계가 있을까?
# Cabin : 선실명 -> 필요없는 변수로 판단됨, 결측값 수 : 690
# Embarked : 승객이 배에 승선한곳, 결측값 수 : 2

#%% Survived
titanic.groupby('Survived').count()
sns.countplot(data = titanic, x = 'Survived')

#%% Pclass & Survived correlationship

# 선실등급별 승객수
test = titanic.groupby('Pclass').count()
sns.countplot(data = titanic, x = 'Pclass')
# 1등급 : 216, 2등급 : 184, 3등급 491

# 선실등급별 생존/사망 승객수
test = titanic.groupby(['Pclass', 'Survived']).count()
sns.countplot(data = titanic, x = 'Pclass', hue = 'Survived')
# 1등급에서 3등급으로 갈수록 생존비율이 현저히 떨어진다.

#%% Name & Survived correlationship

# 포지션 뽑는 함수 생성
def find_position(string):
    position_compile = re.compile('[A-z]*\.')
    position = position_compile.findall(string)
    return position[0][:-1]

# 포지션 열 생성
titanic['position'] = titanic['Name'].apply(lambda x : find_position(x))

# 포지션 빈도수
test = titanic.groupby('position').count()['PassengerId']
sns.countplot(data = titanic, y = 'position')
    # 포지션을 빈도수가 많은 Mr, Mrs, Miss, Master 4개로 축약한다.

test = titanic.groupby(['position', 'Sex']).count()['PassengerId']
sns.countplot(data = titanic, y = 'position', hue = 'Sex')
    # 주포지션을 제외하면 Dr 만 두 성별이 공존하고 나머지는 한가지 성별만 존재한다.

#%% Sex & Survived correlationship
    
test = titanic.groupby(['Sex', 'Survived']).count()
sns.countplot(data = titanic, x = 'Sex', hue = 'Survived')
# 남자는 생존 비율이 매우 낮고 여자는 생존비율이 매우 높다.

#%% Age & Survived correlationship

sns.distplot(titanic[(titanic['Survived'] == 0) & (titanic['Age'].notnull())].loc[:, 'Age'])
sns.distplot(titanic[(titanic['Survived'] == 1) & (titanic['Age'].notnull())].loc[:, 'Age'])

# 어린이(~13, 0), 청소년(13~20, 1), 청년(20~30, 2), 장년(30~45, 3), 중년(45~60, 4), 노년(60~, 5) 으로 구분
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
        return age
    
titanic['aged'] = titanic['Age'].apply(lambda x : get_aged(x))

test = titanic.groupby(['aged', 'Sex', 'Survived']).count()
sns.countplot(data = titanic, x = 'aged', hue = 'Sex')

# 나이대 변수에 따라서 남녀 생존율이 극명하게 갈린다.

#%% SibSp & Survived correlationship

test = titanic.groupby(['SibSp', 'Survived']).count()

# SibSp -> SibSp_existence (형제 자매의 유무 변수로 변경, 0 : 없다, 1 : 있다)
def get_SibSp_existence(SibSp):
    if SibSp >= 1:
        return 1
    else:
        return SibSp

titanic['SibSp_existence'] = titanic['SibSp'].apply(lambda x : get_SibSp_existence(x))
test = titanic.groupby(['SibSp_existence', 'Survived']).count()

# 형제, 사촌이 있으면 생존율이 없는경우에 비해 높다.

#%% Parch & Survived correlationship

test = titanic.groupby(['Parch', 'Survived']).count()


# Parch -> Parch_existence (형제 자매의 유무 변수로 변경, 0 : 없다, 1 : 있다)
def get_Parch_existence(Parch):
    if Parch >= 1:
        return 1
    else:
        return Parch

titanic['Parch_existence'] = titanic['Parch'].apply(lambda x : get_Parch_existence(x))
test = titanic.groupby(['Parch_existence', 'Survived']).count()

# 부모 자식이 있으면 생존율이 없는경우에 비해 높다.

#%% Fare & Survived correlationship

sns.boxplot(data = titanic, x = 'Survived', y = 'Fare')

# 생존자의 평균 운임 요금이 사망자의 평균 운임요금에 비해 높지만
# t-검정을 이용해 통계적으로 유의한지 알아보자. 유의수준은 0.05로 둔다

from scipy import stats
stats.ttest_ind(titanic[titanic['Survived'] == 0]['Fare'], titanic[titanic['Survived'] == 1]['Fare'])
# p-valu가 6.120189341924198e-15로 유의수준보다 낮아 통계적으로 유의하다는것을 확인할 수 있다.

#%% Embarked & Survived correlationship

test = titanic.groupby(['Embarked', 'Survived', 'Sex']).count()
# C 와 Q 에서는 생존자 수와 사망자 수가 크게 차이나지 않지만 S에서는 많이 차이가 난다는 것을 확인 할 수 있다.
