import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as proc
import os

colors = ['r','g','b','c','o']

index = 0
legend = []
for file in os.listdir('.'):    
    if file.endswith('.txt') and 'loss' in file:
        plt.plot(proc.medfilt(np.loadtxt(file),51)[:-20],colors[index])    
        legend.append(file)
        index = index + 1
plt.legend(legend)
plt.show()        
plt.waitforbuttonpress()
