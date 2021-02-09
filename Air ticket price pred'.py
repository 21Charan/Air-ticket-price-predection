import pandas as pd 
import numpy as np 
import sklearn as sks

import os 
os.chdir(r"C:\Users\user\Desktop\projects and code\Hackathons\air ticket price predection")

train  = pd.read_excel("Data_Train.xlsx")
test   = pd.read_excel("Test_set.xlsx")
sample_submission=pd.read_excel("Sample_submission.xlsx")

train['Total_Stops'].replace(['non-stop','2 stops','1 stop','3 stops','4 stops'],[0,2,1,3,4],inplace=True)
train['Total_Stops'].fillna(1, inplace =True)
train['Total_Stops'] = train['Total_Stops'].astype(str)

test['Total_Stops'].replace(['non-stop','2 stops','1 stop','3 stops','4 stops'],[0,2,1,3,4],inplace=True)
test['Total_Stops'].fillna(1, inplace =True)

test['Total_Stops'] = test['Total_Stops'].astype(str)

def f(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12 ):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return'Noon'
    elif (x > 16) and (x <= 20) :
        return 'Eve'
    elif (x > 20) and (x <= 24):
        return'Night'
    elif (x <= 4):
        return'Late Night'

train['Dep_Time'] = train.Dep_Time.str.split(':', expand=True)[0]
train['Dep_Time'] = pd.to_numeric(train['Dep_Time'])
train['Dep_Time'] = train['Dep_Time'].apply(f)

test['Dep_Time'] = test.Dep_Time.str.split(':', expand=True)[0]
test['Dep_Time'] = pd.to_numeric(test['Dep_Time'])
test['Dep_Time'] = test['Dep_Time'].apply(f)

train['Duration']=  train['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
test['Duration']=  test['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)

train['Date_of_Journey'] = pd.to_datetime(train['Date_of_Journey'])
train['month'] = train['Date_of_Journey'].dt.month
train['Day'] = train['Date_of_Journey'].dt.day
train['dayofweek'] = pd.to_datetime(train['Date_of_Journey']).dt.dayofweek

test['Date_of_Journey'] = pd.to_datetime(test['Date_of_Journey'])
test['month'] = test['Date_of_Journey'].dt.month
test['Day'] = test['Date_of_Journey'].dt.day
test['dayofweek'] = pd.to_datetime(test['Date_of_Journey']).dt.dayofweek


train['Airline'] = pd.factorize(train['Airline'])[0]
train['Dep_Time'] = pd.factorize(train['Dep_Time'])[0]
train['Total_Stops'] = pd.factorize(train['Total_Stops'])[0]

test['Airline'] = pd.factorize(test['Airline'])[0]
test['Dep_Time'] = pd.factorize(test['Dep_Time'])[0]
test['Total_Stops'] = pd.factorize(test['Total_Stops'])[0]

train=train.drop(['Date_of_Journey','Route','Arrival_Time','Additional_Info','Source','Destination'],axis=1)
test=test.drop(['Date_of_Journey','Route','Arrival_Time','Additional_Info','Source','Destination'],axis=1)

def outlier(df):
    for i in df.describe().columns:
        Q1=df.describe().at['15%',i]
        Q3=df.describe().at['85%',i]
        IQR= Q3-Q1
        LE=Q1-1.5*IQR
        UE=Q3+1.5*IQR
        df[i]=df[i].mask(df[i]<LE,LE)
        df[i]=df[i].mask(df[i]>UE,UE)
    return df

train=outlier(train)
test=outlier(test)

def onehotencode(data):
    cat=data.select_dtypes(include=['object'])
    list_cat=list(cat.columns)
    dataf = pd.concat([data.drop(list_cat, axis=1), pd.get_dummies(cat)], axis=1)
    return dataf

train=onehotencode(train)
test=onehotencode(test)

from sklearn.preprocessing import scale
data_standardized=scale(train)
data_standardized.mean(axis=0)

#Take targate variable into y
y = train['Price']
X = train.drop('Price',axis = 1)

from sklearn.preprocessing import scale
data_standardized=scale(test)
data_standardized.mean(axis=0)


# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X,y)

import xgboost
xgb=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
colsample_bytree=1, max_depth=7)

xgb.fit(X_train,y_train)

from sklearn.linear_model import LinearRegression
lr_model=model = LinearRegression()

lr_model.fit(X, y)

from sklearn.neighbors import KNeighborsRegressor
# instantiate the model and set the number of neighbors to consider to 3
knn = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
knn.fit(X,y)

# Predicting results for test dataset
y_pred = rf.predict(test)
submission = pd.DataFrame({"Price": y_pred})
submission.to_csv('submission_rf_bbb.csv', index=False)

# Predicting results for test dataset
y_pred = lr_model.predict(test)
submission = pd.DataFrame({"Price": y_pred})
submission.to_csv('submission_lr_aaaa.csv', index=False)

# Predicting results for test dataset
y_pred = xgb.predict(test)
submission = pd.DataFrame({"Price": y_pred})
submission.to_csv('submission_xgb_aaaa.csv', index=False)

# Predicting results for test dataset
y_pred = knn.predict(test)
submission = pd.DataFrame({"Price": y_pred})
submission.to_csv('submission_knn.csv', index=False)
