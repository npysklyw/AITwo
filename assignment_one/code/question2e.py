import numpy as np
import matplotlib.pyplot as plt

#Load train data
x1 = np.loadtxt( 'hw1xtr.dat' )
y1 = np.loadtxt( 'hw1ytr.dat' )

#Load test data
x2 = np.loadtxt( 'hw1xte.dat' )
y2 = np.loadtxt( 'hw1yte.dat' )

#Add cubic, quadratic terms in the data, also provide the column of ones
modifiedx1 = np.c_[np.power(x1,3), np.power(x1,2),x1, np.ones(40) ]    
modifiedx2 = np.c_[np.power(x2,3),np.power(x2,2),x2, np.ones(20) ]   

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

#Make it easier to plot the graph
#If we do we what we did in q2b and c, the plot of the polynomial model is very messy
polyModel = np.poly1d(list(weights))
samplepoints = np.linspace(0,x1.max(),40)

#Plot the train data, with the model overlayed
plt.title("Train Data with 3th-order polynomial regression")
plt.ylabel('target')
plt.xlabel('feature')
plt.scatter(x1,y1)
plt.plot(samplepoints,polyModel(samplepoints),'r')
plt.show()

#Plot the test data, with the model overlayed
samplepoints2 = np.linspace(0,x2.max(),20)
plt.title("Test Data with 3th-order polynomial regression")
plt.ylabel('target')
plt.xlabel('feature')
plt.scatter(x2,y2)
plt.plot(samplepoints2,polyModel(samplepoints2),'r')
plt.show()

#Report the error values
print("Train Error",averageError(weights,modifiedx1.T,y1))
print("Test Error",averageError(weights,modifiedx2.T,y2))