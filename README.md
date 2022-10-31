# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Start the program
2.Import the numpy.pandas and matplotlib
3.Read the file which store the data
4.Declare x as hours and y as scores of the data
5.Using loop predit the data and find the y-intercept, slope using the formulae.
6.Find the best fit using the straight line formula
7.Display the data in graph using the matplotlib libraries
8.Stop the Program.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: prathima
RegisterNumber:  212220040156
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city(10,000s")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")
plt.show()
def computecost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computecost(x,y,theta)
def gradientdescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    j_history.append(computecost(x,y,theta))
  return theta,j_history
theta,j_history=gradientdescent(x,y,theta,0.01,1500)
print("h(x) = "+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)")
plt.title("Cost function using Gradient Descent")
plt.show()
plt.scatter(data[0],data[1],color="black")
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="red")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
plt.show()
def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))
*/
```

## Output:
![image](https://github.com/prathima2002/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/b98f95a15d54775da8aade71d4223f854a94b123/WhatsApp%20Image%202022-10-31%20at%2009.39.25.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/42ba373b32ba01cd6439d918d3eaa265f1d4d5b3/WhatsApp%20Image%202022-10-31%20at%2009.40.03.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/c55268f73b68a6ce4487a4d7ff3b539c0ba3df35/WhatsApp%20Image%202022-10-31%20at%2009.40.38.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/f159bfacaca06c237e1d0ee6393d18c0be39db89/WhatsApp%20Image%202022-10-31%20at%2009.41.02.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/98e77988c96dc99bcc2607d7b44fc9725d9ca651/WhatsApp%20Image%202022-10-31%20at%2009.42.02.jpeg)

![image](https://github.com/prathima2002/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/62cd1136778b1ece1d287ea05dbddcd8d97ab753/WhatsApp%20Image%202022-10-31%20at%2009.42.27.jpeg)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
