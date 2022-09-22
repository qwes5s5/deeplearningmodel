import numpy as np 
import matplotlib.pyplot as plt 

num_of_intervals = 2000
x = np.linspace(-10,10,num=num_of_intervals)

y_inputs = 1/(1+np.exp(-x)) # SIGMOID FUNCTION

plt.figure(figsize = (15,9))
plt.plot(x,y_inputs,label = 'Sigmoid Function')
plt.vlines(x=0, ymin = min(y_inputs), ymax=max(y_inputs), linestyles='dashed')
plt.title('Sigmoid Function')
plt.show()