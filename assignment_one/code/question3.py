import numpy as np
import matplotlib.pyplot as plt

#Load train data
x1 = np.loadtxt( 'hw1xtr.dat' )
y1 = np.loadtxt( 'hw1ytr.dat' )

#Load test data
x2 = np.loadtxt( 'hw1xte.dat' )
y2 = np.loadtxt( 'hw1yte.dat' )

#List of the regularization values we wish to test
lambdaValues = [0.01,0.05,0.1,0.5,1,100,1000000]

#Add quartic, cubic, quadratic terms in the data, also provide the column of ones
modifiedx1 = np.c_[np.power(x1,4),np.power(x1,3), np.power(x1,2),x1, np.ones(40) ]    
modifiedx2 = np.c_[np.power(x2,4),np.power(x2,3),np.power(x2,2),x2, np.ones(20) ]   

#function h(x)
def hypothesis(x,w):
    return np.matmul(x,w)

def gradientDescent(X,y,currentLamba):
    return np.linalg.inv(np.matmul(X.T,X) + currentLamba*np.identity(5))@X.T@y

#Function to calculate the average error based on weight, x, and y
def averageError(w,x,y):
    return sum((hypothesis(w,x) - y)*((hypothesis(w,x) - y)))/x.shape[1]

#All used for questions a,b in the plotting
trainerror = []
testerror = []
weightArray = []

#Iteratively traverse the lambda values
for value in lambdaValues:

    #Get the weight values, calculated with new formula with lambda
    weights = gradientDescent(modifiedx1,y1,np.log(value))

    #Save weights
    weightArray.append(weights)

    #Calculate and save the average train, test errors
    trainerror.append(averageError(weights,modifiedx1.T,y1 ))
    testerror.append(averageError(weights,modifiedx2.T,y2))

#Plot 
plt.title("Train/Test Error over Lambda")
plt.plot(lambdaValues,trainerror, marker="o",label="train")
plt.plot(lambdaValues,testerror, marker="o",label="test")
plt.legend()
plt.xscale("log")
plt.grid(alpha=0.6)
plt.ylabel('error')
plt.xlabel('lambda')
plt.show()

plt.title("Value of Each Weight Parameter as a Function of Lambda")

#Used to plot the weight values in an organized manner for question b
windex = [[],[],[],[],[]]
for w in weightArray:
    for x in range(0,5):
        windex[x].append(w[x])

#Plot each weight parameters change over lambda
for x in range(0,5):
    plt.plot(lambdaValues,windex[x],label="weight" + str(x))

#Add the plot parameters
plt.legend()
plt.xscale("log")
plt.grid(alpha=0.6)
plt.ylabel('weight value')
plt.xlabel('lambda')
plt.show()






# def gradientDescent2(X,y,currentLamba):
#     print(X.shape)
#     print(y)
#     return np.linalg.inv(np.matmul(X.T,X) + currentLamba*np.identity(5))@X.T@y


# def fivefoldCV(lambdas):
#     #leave out a specific set
#     train2 = np.array_split(modifiedx1, 5)
#     train2y = np.array_split(y1, 5)
#     errors = []
#     list = [1,2,3,4,5]
#     for value in lambdas:
#         for x in range(1,6):
#             data = []
#             datay = []
#             validation = train2[x-1] 
#             validationy = train2y[x-1] 
#             list.remove(x)

#             for nt in list:
#                 data.append(train2[x-1])
#                 datay.append(train2y[x-1])
#             print(np.array(data).T)
#             print(np.array(datay))
#             weights = gradientDescent2(np.array(data).T,np.array(datay),value)
#             print(averageError(weights,np.array(validation).T,np.array(validationy)))
        
#             list = [1,2,3,4,5]




# fivefoldCV(lambdaValues)