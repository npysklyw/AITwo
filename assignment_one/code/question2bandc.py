import numpy as np
import matplotlib.pyplot as plt

#Load train data
x1 = np.loadtxt( 'hw1xtr.dat' )
y1 = np.loadtxt( 'hw1ytr.dat' )

#Load test data
x2 = np.loadtxt( 'hw1xte.dat' )
y2 = np.loadtxt( 'hw1yte.dat' )

#Add a column of ones to data
modifiedx1 = np.c_[ x1, np.ones(40) ]    

#function h(x)
def hypothesis(x,w):
    return np.matmul(x,w)


def gradientDescent(X,y):
    return np.linalg.inv(np.matmul(X.T,X))@X.T@y

#Function to calculate the average error based on weight, x, and y
def averageError(w,x,y):
    return sum((hypothesis(w,x) - y)*((hypothesis(w,x) - y)))/x.shape[1]

#Get the weight values
weights= gradientDescent(modifiedx1,y1)

#Plot the train data, alongside the linear regression model
plt.title("Train Data with Linear Regression")
plt.ylabel('target')
plt.xlabel('feature')
plt.plot(x1,y1, 'yo', x1, hypothesis(modifiedx1,weights)) 
plt.show()

#Ensure the test data is properly set up to use with our linear regression model
modifiedx2 = np.c_[ x2, np.ones(20) ]

#Plot the test data, alongside the linear regression model
plt.title("Test Data with Linear Regression")
plt.ylabel('target')
plt.xlabel('feature')
plt.plot(x2,y2, 'yo', x2, hypothesis(modifiedx2,weights)) 
plt.show()

#Report train & test error
print("Train Error: ",averageError(weights,modifiedx1.T,y1))
print("Test Error: ",averageError(weights,modifiedx2.T,y2))