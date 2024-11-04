# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Jabez S
RegisterNumber:  212223040070
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

# Output:
## Encoding:
![1](https://github.com/user-attachments/assets/ab11516c-0e8a-429f-87ff-a4caf17ddfb7)
## Head():
![2](https://github.com/user-attachments/assets/f52af709-69ae-46a8-a6a0-089de0259bdf)
## Info():
![3](https://github.com/user-attachments/assets/11d09044-e33a-4e74-a776-35927fa3219b)
## isnull().sum():
![4](https://github.com/user-attachments/assets/fdc99a88-4db0-4a4b-b71a-6356e66a782f)
## Prediction of y:
![5](https://github.com/user-attachments/assets/e0edea41-b976-4d6a-b248-a0ec8000366e)
## Accuracy:
![6](https://github.com/user-attachments/assets/764ba3f5-0a30-44b7-99d3-c29c585344d6)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
