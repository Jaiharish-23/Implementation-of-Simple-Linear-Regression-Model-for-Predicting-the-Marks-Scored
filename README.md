# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: JAI HARISH R

RegisterNumber: 212224040124

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

df.tail()

x = df.iloc[:,:-1].values
print(x)

y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

print(y_pred)

print(y_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='violet')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
head:

![{1ABD0804-2E97-4669-9B4F-3C0942DE3D2D}](https://github.com/user-attachments/assets/39e38f1b-56e2-4fd7-b22d-c15bcf434ab8)

tail:

![{E8A6E4FC-30CB-44B5-9C78-04525E132782}](https://github.com/user-attachments/assets/83c13f2e-721f-4c2a-b34d-809a00aa36d6)


Segregating data to variables:

![{51E03067-B3B9-432D-8BCF-A7E09DAFF397}](https://github.com/user-attachments/assets/091a1829-6201-4aaf-96de-bcf0d33ac744)

![{3634D752-10A5-4F52-8615-E9F40CFE58DD}](https://github.com/user-attachments/assets/ce8608cc-3a45-44cc-98bf-6f960168f264)

Displaying predicted values:

!![{3FFF12D5-06DC-4FEA-81A3-4E89F8D2D008}](https://github.com/user-attachments/assets/3f5ae6f8-f94f-49d9-8b28-1f1219e74779)


Displaying actual values:

![{281BAB57-8207-4EB9-9101-2C7172123E3A}](https://github.com/user-attachments/assets/fba7c58c-b9cb-444a-99ac-20c977cfabff)

Graph plot for training data:

![{33CD33BE-820D-4226-AF99-9A7C611220BB}](https://github.com/user-attachments/assets/d8fc522b-e564-4966-920a-e3800b771b3b)

Graph plot for test data:

![{BB4B8BF1-3E60-4C77-97A4-95320E62AA13}](https://github.com/user-attachments/assets/f4785621-e29d-4677-a08e-467def4f54b3)

MSE MAE RMSE:

![{F85A7C57-B9CE-464A-A627-931B97BC9082}](https://github.com/user-attachments/assets/49ea0a67-859a-4638-b63b-0824b3702baa)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
