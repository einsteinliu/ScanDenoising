import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as proc

loss = np.loadtxt('loss.txt')
loss = proc.medfilt(loss,101)
plt.plot(loss)
plt.show()
plt.waitforbuttonpress()
