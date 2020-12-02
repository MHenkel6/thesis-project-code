# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:05:46 2020

@author: Martin
"""

import numpy as np
import pandas as pd 
import os
import glob
rootPath = "D:/Files/TUDelftLocal/Thesis/Software/Simulation/"
filePath = rootPath + "DataScenario/"
suffix = "HighNoiseV3"
globPattern = "Kalman"+suffix+"_*" # CentralizedKalmanFilterEval_*.csv"
fileList = glob.glob(filePath+globPattern)
datList = [] 
counter = 0
maxLen = len(fileList)
for file in fileList:
    if not os.path.isfile(file):
        print(file+" is missing")
    else:
        npdat = np.genfromtxt(file,delimiter=',')
        fSat = int(npdat[0])
        fThr = int(npdat[1])
        fType = int(npdat[2])
        fParam = npdat[3]
        fTime = npdat[4]
        detTime = npdat[5]
        tLen = len(npdat)-6
        faultVector = npdat[6:6+tLen//2].astype(int)
        
        if suffix == "MultiFail":
            isoVector = npdat[6+6+tLen//2:6+6+tLen].astype(int)
            isoDetailed = npdat[6+tLen:].astype(float)
            datList.append([fSat ,fThr ,fType ,fParam ,fTime ,detTime ,faultVector ,isoVector,isoDetailed])
        else:
            isoVector = npdat[6+tLen//2:].astype(int)
            datList.append([fSat ,fThr ,fType ,fParam ,fTime ,detTime ,faultVector ,isoVector])
        
if suffix == "MultiFail":
    cols = ['fSat','fThr','fType','fParam','fTime','detTime','faultVector',
            'isoVector','isoDetailed']
else:
    cols = ['fSat','fThr','fType','fParam','fTime','detTime','faultVector',
          'isoVector']
df = pd.DataFrame(data = datList, 
                  columns = ['fSat','fThr','fType','fParam','fTime','detTime','faultVector',
                            'isoVector',])
fileName = "CentralizedKalmanCombined_"+suffix+".hdf"
df.to_hdf(filePath+fileName,key = 'DataKalman',mode='w')