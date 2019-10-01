# importing the required module 
import matplotlib.pyplot as plt
import numpy as np
  
y = np.loadtxt("/Users/zhutao/work/CPEN502/trainError.txt")

# plotting the points 
plt.plot(y) 

# naming the x axis 
plt.xlabel('epoch') 
# naming the y axis 
plt.ylabel('total error') 

# giving a title to my graph
plt.title('figure c-1) binary, momentum = 0.9') 

# function to show the plot 
plt.show() 
