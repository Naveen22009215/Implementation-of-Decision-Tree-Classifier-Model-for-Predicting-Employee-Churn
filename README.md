# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Libraries.

2.Upload the dataset in the compiler and read the dataset.

3.Find head,info and null elements in the dataset.

4.Using LabelEncoder and DecisionTreeClassifier , find accuracy and prediction for the dataset.

5.End the program.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: P NAVEEN KUMAR
RegisterNumber:  212222230092
*/
```
```

import pandas as pd
data=pd.read_csv("/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```


## Output:
## 1. data.head()
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119401470/3247917a-81c9-4179-a67b-67b3cbd229e8)

## 2. data.info()
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119401470/c1d08b49-6158-4d7d-9199-d56895d8ae11)

## 3. isnull() and sum()
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119401470/e8228dba-f7ca-4b27-9d12-61d6bffa9842)

## 4. data value counts()
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119401470/1b3efcfb-14cb-463c-ba84-fc80000e57ac)

## 5. data.head() for salary
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119401470/7afdcb07-343a-4a6c-83fd-1c9612c78305)

## 6. x.head()
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119401470/86cf44f0-46cf-4471-9c2e-8f8989471088)

## 7. accuracy value
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119401470/42e8ca08-8aac-4500-a3b9-32cdc8b31815)

## 8. data prediction
![image](https://github.com/Naveen22009215/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119401470/40cfe63a-9afd-421c-97db-90b08b21052e)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
