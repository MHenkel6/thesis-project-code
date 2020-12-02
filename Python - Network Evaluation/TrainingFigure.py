# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:48:07 2020

@author: Martin
"""

import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

epochs = np.arange(1,18)
trainArrayAcc = np.array([0.5308,0.5669,0.5617,0.5961,0.5934,
                          0.6052,0.6022,0.6107,0.6193,0.6178,
                          0.6284,0.6303,0.6363,0.6455,0.6331,
                          0.6489,0.6431])
trainArrayLoss = np.array([1.6240,1.4828,1.7508,1.3963,1.3787,
                           1.3386,1.3900,1.3716,1.3157,1.3066,
                           1.2632,1.2372,1.2058,1.2192,1.2334,
                           1.2062, 1.1976])

validArrayAcc = np.array([0.5556,0.5723,0.5731,0.6079,0.6006,
                          0.6243,0.6141,0.6187,0.6375,0.6441,
                          0.6484,0.6497,0.6420,0.6634,0.6466,
                          0.6647,0.6543])
validArrayLoss = np.array([1.5490,1.4328,1.6842,1.3456,1.3292,
                           1.2651,1.3337,1.3098,1.2197,1.2178,
                           1.2020,1.1784,1.1603,1.1587,1.1740,
                           1.1646, 1.1715])

lwidth = 3
fig, ax = plt.subplots()
plt.minorticks_on()
ax.plot(epochs,100*trainArrayAcc,linewidth = lwidth,color = '#d62728', label =  "Categorical Accuracy on Training Set")
ax.plot(epochs,100*validArrayAcc,linewidth = lwidth,color = '#ff7f0e', label =  "Categorical Accuracy on Validation Set")
ax.set_ylabel("Categorical Accuracy [%]")
# ax.set_title("Individual Isolation Network Progress over Training Period")
ax.grid(True, which = "both")
ax.set_ylim([50,70])
plt.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

fig, ax = plt.subplots()
plt.minorticks_on()        
ax.plot(epochs,trainArrayLoss, linewidth = lwidth, color = '#1f77b4', label =  "Categorical Cross-Entropy on Validation Set")
ax.plot(epochs,validArrayLoss, linewidth = lwidth, color = '#17becf', label =  "Categorical Cross-Entropy on Validation Set")
ax.set_ylabel("Categorical Cross-Entropy [-]")
ax.grid(True, which = "both")
ax.set_xlabel("Number of Training Epoch")
ax.set_ylim([1.2,2])
plt.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()