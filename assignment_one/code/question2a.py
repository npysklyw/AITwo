import numpy as np
import matplotlib.pyplot as plt

#Load train data
x1 = np.loadtxt( 'hw1xtr.dat' )
y1 = np.loadtxt( 'hw1ytr.dat' )

#Load test data
x2 = np.loadtxt( 'hw1xte.dat' )
y2 = np.loadtxt( 'hw1yte.dat' )

#Plot train data
plt.scatter(x1,y1)
plt.title('Train Data')
plt.ylabel('hw1ytr - target')
plt.xlabel('hw1xtr  - feature')
plt.show()

#Plot test data
plt.scatter(x2,y2)
plt.title('Test Data')
plt.ylabel('hw1yte - target')
plt.xlabel('hw1xte  - feature')
plt.show()