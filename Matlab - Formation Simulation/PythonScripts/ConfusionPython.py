# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:03:56 2020
Script to evaluate the Kalman Data
@author: Martin
"""
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')
#%%
def plot_confusion_matrix(cm,y_true,y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, 
                          annotate=True,
                          maximize = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    
    # Only use the labels that appear in the data
    classes = [classes[x] for x in unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    
    # Lines and size
    for pos in np.arange(0.5, len(classes)+0.5):
      ax.axhline(pos, linestyle='--')
      ax.axvline(pos, linestyle='--')
    fig.set_size_inches(15,15)
    
    if annotate:
      # Loop over data dimensions and create text annotations.
      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i in range(cm.shape[0]):
          for j in range(cm.shape[1]):
              ax.text(j, i, format(cm[i, j], fmt),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black",
                      size=7)
    fig.tight_layout()
    if maximize:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    plt.show()
    return ax
filePath = "DataScenario/"
figurePath = "figures/ThesisMainFigures/"
fileSuffix = "LowIntensity"
KalmanData = pd.read_hdf(filePath+"CentralizedKalmanCombined_"+ fileSuffix +".hdf")
print(KalmanData['fSat'].dtype)
print(KalmanData['fThr'].dtype)
print(KalmanData['fType'].dtype)
print(KalmanData['fParam'].dtype)
print(KalmanData['fTime'].dtype)
print(KalmanData['detTime'].dtype)
print(KalmanData['faultVector'].dtype)
print(KalmanData['isoVector'].dtype)
#%% Plot Confusion Matrix
save = False

boolClosedFault = KalmanData['fType'] == 1
boolOpenFault = KalmanData['fType'] == 2

# Keys of data are ['fSat','fThr','fType','fParam','fTime','detTime','faultVector',isoVector']
faultCombined = np.concatenate(KalmanData['faultVector'].to_numpy())
isoCombined = np.concatenate(KalmanData['isoVector'].to_numpy())
detCombined = np.ceil(np.divide(isoCombined,36))
faultDet =  np.ceil(np.divide(faultCombined,36))

isoPair = np.ceil(np.divide(isoCombined,2))
faultPair = np.ceil(np.divide(faultCombined,2))

cmDet = confusion_matrix(faultDet,detCombined)
cmIso = confusion_matrix(faultCombined,isoCombined)
cmPair = confusion_matrix(faultPair,isoPair)
#%% Plot Comfusion Matrix
detClasses = {0:"Faultless",1:"Faulty"}
isoclasses = {0:'Faultless', 
              1: "S1, T1",2: "S1, T2",3: "S1, T3",4: "S1, T4",5: "S1, T5",6: "S1, T6",
              7: "S2, T1",8: "S2, T2",9: "S2, T3",10:"S2, T4",11:"S2, T5",12:"S2, T6",
              13:"S3, T1",14:"S3, T2",15:"S3, T3",16:"S3, T4",17:"S3, T5",18:"S3, T6",
              19:"S4, T1",20:"S4, T2",21:"S4, T3",22:"S4, T4",23:"S4, T5",24:"S4, T6",
              25:"S5, T1",26:"S5, T2",27:"S5, T3",28:"S5, T4",29:"S5, T5",30:"S5, T6",
              31:"S6, t1",32:"S6, T2",33:"S6, T3",34:"S6, T4",35:"S6, T5",36:"S6, T6",}
pairClasses = {0:'Faultless', 
              1: "S1, TP1",2: "S1, TP2",3: "S1, TP3",4: "S2, TP1",5: "S2, TP2",6: "S2, TP3",
              7: "S3, TP1",8: "S3, TP2",9: "S3, TP3",10:"S4, TP1",11:"S4, TP2",12:"S4, TP3",
              13:"S5, TP1",14:"S5, TP2",15:"S5, TP3",16:"S6, TP1",17:"S6, TP2",18:"S6, TP3"}

plot_confusion_matrix(cmDet,faultDet,detCombined,detClasses,normalize =True,title= "Kalman Fault Detection Confusion Matrix")
if save:
    plt.savefig(figurePath + "Kalman"+ fileSuffix +"FaultDetection_ConfusionMatrix.pdf")
plot_confusion_matrix(cmIso,faultCombined,isoCombined,isoclasses,normalize =True,title= "Kalman Fault Isolation Confusion Matrix",maximize = True)
if save:
    plt.savefig(figurePath + "KalmanFaultIsolation_ConfusionMatrix.pdf")
plot_confusion_matrix(cmPair,faultPair,isoPair,pairClasses,normalize =True,title= "Kalman Thruster Pair (TP) Isolation Confusion Matrix",maximize = True)
if save:
    plt.savefig(figurePath + "Kalman"+ fileSuffix +""+ fileSuffix +"FaultPairIsolation_ConfusionMatrix.pdf")

# Determine fault detection delay vs intensity per fault
detDelayClosed = KalmanData[boolClosedFault]['detTime'].to_numpy()-KalmanData[boolClosedFault]['fTime'].to_numpy()
detDelayOpen = KalmanData[boolOpenFault]['detTime'].to_numpy()-KalmanData[boolOpenFault]['fTime'].to_numpy()

paramsClosed = KalmanData[boolClosedFault]['fParam'].to_numpy()
paramsOpen = KalmanData[boolOpenFault]['fParam'].to_numpy()


#%% Determine average reaction time per fault over a 0.1 fault intensity interval
delayClosed = np.zeros([9,1])
delayOpen = np.zeros([9,1])
for i in range(0,9):
    para = 0.1 + i/10
    boolPara = (KalmanData['fParam'] >para) & (KalmanData['fParam'] < para + 0.1)
    tempDelayClosed = KalmanData[boolPara & boolClosedFault]['detTime'].to_numpy()-KalmanData[boolPara & boolClosedFault]['fTime'].to_numpy()
    tempDelayOpen = KalmanData[boolPara & boolOpenFault]['detTime'].to_numpy()-KalmanData[boolPara & boolOpenFault]['fTime'].to_numpy()
    
    delayClosed[i] = np.mean(tempDelayClosed[tempDelayClosed>0])
    delayOpen[i] = np.mean(tempDelayOpen)
    

#%% Plotting Delay Time
boxLabels = ["[0.1,0.2)","[0.2,0.3)","[0.3,0.4)","[0.4,0.5)",
             "[0.5,0.6)","[0.6,0.7)","[0.7,0.8)","[0.8,0.9)",
             "[0.9,1.0]"]
    
plt.figure(figsize=[1.25*6.4, 1.25*4.8])
detDelayClosed_Plot = detDelayClosed
detDelayClosed_Plot[detDelayClosed < 0] = np.nan
df = pd.DataFrame(np.reshape(detDelayClosed_Plot,[9,len(detDelayClosed_Plot)//9]).transpose(),columns = boxLabels)
sns.boxplot(data=df,showfliers = False,color="royalblue")
plt.title("Distribution of Closed Fault Detection Time")
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity Interval [-]")
bottom, top = plt.ylim()  
#plt.ylim([0,top])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.show()
if save:
    plt.savefig(figurePath + "KalmanClosed"+ fileSuffix +"_DetectionTime_BoxPlot.pdf")

plt.figure(figsize=[1.25*6.4, 1.25*4.8])
df = pd.DataFrame(np.reshape(detDelayOpen,[9,len(detDelayOpen)//9]).transpose(),columns = boxLabels)
sns.boxplot(data=df,showfliers = False,color="royalblue")
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity Interval [-]")
plt.title("Distribution of Open Fault Detection Time")
bottom, top = plt.ylim()  
#plt.ylim([0,top])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.show()
if save:
    plt.savefig(figurePath + "KalmanOpen"+ fileSuffix +"_DetectionTime_BoxPlot.pdf")

#%% Plot Reaction time vs Intensity 
steps = 100
delayClosedFine = np.zeros([steps,1])
delayOpenFine = np.zeros([steps,1])
paraArray = np.linspace(0.1,1,steps)
for i in range(0,steps):
    para = 0.1 + i*0.9/steps
    boolPara = (KalmanData['fParam'] > para - 0.00001) & (KalmanData['fParam'] < para + 0.9/steps + 0.00001)
    tempDelayClosed = KalmanData[boolPara & boolClosedFault]['detTime'].to_numpy()-KalmanData[boolPara & boolClosedFault]['fTime'].to_numpy()
    tempDelayOpen = KalmanData[boolPara & boolOpenFault]['detTime'].to_numpy()-KalmanData[boolPara & boolOpenFault]['fTime'].to_numpy()
    delayClosedFine[i] = np.nanmean(tempDelayClosed[tempDelayClosed>0])
    delayOpenFine[i] = np.mean(tempDelayOpen)
closedTrend = np.polyfit(paraArray,delayClosedFine,1)
openTrend = np.polyfit(paraArray,delayOpenFine,1)

plt.figure()
plt.plot(paraArray,delayClosedFine,'b',label='Averaged Detection Time')
plt.plot(paraArray,closedTrend[1]+closedTrend[0]*paraArray,'r',label='Trend line')
plt.title("Averaged Closed Fault Detection Time")
plt.ylabel("Averaged Detection Time [s]")
plt.xlabel("Fault Intensity [-]")
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.legend()
plt.show()
if save:
    plt.savefig(figurePath + "KalmanClosed"+ fileSuffix +"_DetectionTime_Linear.pdf")

plt.figure()
plt.plot(paraArray,delayOpenFine,'b',label='Averaged Detection Time')
plt.plot(paraArray,openTrend[1]+openTrend[0]*paraArray,'r',label='Trend line')
plt.title("Averaged Open Fault Detection Time")
plt.ylabel("Averaged Detection Time [s]")
plt.xlabel("Fault Intensity [-]")
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.legend()
plt.show()
if save:
    plt.savefig(figurePath + "KalmanOpen"+ fileSuffix +"_DetectionTime_Linear.pdf")
