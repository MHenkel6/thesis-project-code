# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:03:56 2020
Script to evaluate the Kalman data
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
def plot_confusion_matrix2(cm,y_true,y_pred, classes,
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
def plot_confusion_matrix(cm,y_true,y_pred, classes,
                          normalize= None,
                          title=None,
                          cmap=plt.cm.Blues, 
                         annotate=True,
                         textsize = 7,
                         colorbar = True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
  
    # Only use the labels that appear in the data
    classes = [classes[x] for x in unique_labels(y_true, y_pred)]
    if normalize == "row":
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Row normalized confusion matrix")
    elif normalize == "col":
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis,:]
        print("Column normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    fig, ax = plt.subplots()
    if normalize:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap,vmin = 0, vmax = 1)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if colorbar:
        ax.figure.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
   
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Change ticks and axis label font size
    ax.tick_params(axis='both', which='major', labelsize=textsize)
    ax.xaxis.label.set_size(textsize)
    ax.yaxis.label.set_size(textsize)
    
    #fig.tight_layout()
    
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
                      size=textsize)
    fig.tight_layout()
    if title:
        fig.suptitle(title,y = 0.97,fontsize = textsize + 15)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    return ax
def running_mean(x, N):
    if N == 1:
        return x
    else:
        cumsum = np.nancumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
#%%
filePath = "D:\\Files\\TUDelftLocal\\Thesis\\Software\\Simulation\\DataKalman\\"
figurePath = "figures/ThesisMainFigures/"
fileSuffix = ""
KalmanData = pd.read_hdf(filePath+"CentralizedKalman_Combined.hdf")
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
faultCombined = np.concatenate(KalmanData['faultVector'].copy().to_numpy())
isoCombined = np.concatenate(KalmanData['isoVector'].copy().to_numpy())
detCombined = np.ceil(np.divide(isoCombined,36))
faultDet =  np.ceil(np.divide(faultCombined,36))

isoCorrect = isoCombined[isoCombined>0] == faultCombined[isoCombined>0]
isoPair = np.ceil(np.divide(isoCombined,2))
faultPair = np.ceil(np.divide(faultCombined,2))
# Evaluate Accuracy
acc = sum(isoCorrect)/len(isoCorrect)

# Isolation Accuracy for closed faults 
outputCombinedIsoClosed = np.concatenate(KalmanData[boolClosedFault]['isoVector'].to_numpy())
faultCombinedIsoClosed = np.concatenate(KalmanData[boolClosedFault]['faultVector'].to_numpy())
isoCorrectClosed = outputCombinedIsoClosed[outputCombinedIsoClosed>0] == faultCombinedIsoClosed[outputCombinedIsoClosed>0]
closedAcc = sum(isoCorrectClosed)/len(isoCorrectClosed)   
print("Kalman Isolation Accuracy Closed Fault:", np.round(100*closedAcc,2))
cmClosed = confusion_matrix(faultCombinedIsoClosed[outputCombinedIsoClosed>0],outputCombinedIsoClosed[outputCombinedIsoClosed>0])
cmRow = cmClosed.astype('float') / cmClosed.sum(axis=1)[:, np.newaxis]
isoRecClosed = np.nanmean(np.diag(cmRow))
cmCol = cmClosed.astype('float') / cmClosed.sum(axis=0)[np.newaxis,:]
isoPrecClosed = np.nanmean(np.diag(cmCol))
print("Kalman Isolation Precision Closed Fault:", np.round(100*isoPrecClosed,2))
print("Kalman Isolation Recall Closed Fault:", np.round(100*isoRecClosed,2))

# Isolation Accuracy for open faults 
outputCombinedIsoOpen = np.concatenate(KalmanData[boolOpenFault]['isoVector'].to_numpy())
faultCombinedIsoOpen = np.concatenate(KalmanData[boolOpenFault]['faultVector'].to_numpy())
isoCorrectOpen = outputCombinedIsoOpen[outputCombinedIsoOpen>0] == faultCombinedIsoOpen[outputCombinedIsoOpen>0]
openAcc = sum(isoCorrectOpen)/len(isoCorrectOpen)   
print("Kalman Isolation Accuracy Open Fault:", np.round(100*openAcc,2))
cmOpen = confusion_matrix(faultCombinedIsoOpen[outputCombinedIsoOpen>0],outputCombinedIsoOpen[outputCombinedIsoOpen>0])
cmRow = cmOpen.astype('float') / cmOpen.sum(axis=1)[:, np.newaxis]
isoRecOpen = np.nanmean(np.diag(cmRow))
cmCol = cmOpen.astype('float') / cmOpen.sum(axis=0)[np.newaxis,:]
isoPrecOpen = np.nanmean(np.diag(cmCol))
print("Kalman Isolation Precision Open Fault:", np.round(100*isoPrecOpen,2))
print("Kalman Isolation Recall Open Fault:", np.round(100*isoRecOpen,2))

# Pair Isolation
pairIsoCorrect = isoPair[isoPair>0] == faultPair[isoPair>0]
pairAcc = sum(pairIsoCorrect)/len(pairIsoCorrect)
print("Kalman Pair Isolation Accuracy: ", 100*pairAcc)
# Calculte Confusion Matrices
cmDet = confusion_matrix(faultDet,detCombined)
cmIso = confusion_matrix(faultCombined,isoCombined)
cmPair = confusion_matrix(faultPair,isoPair)

tn = cmDet[0,0]
tp = cmDet[1,1]
fp = cmDet[0,1]
fn = cmDet[1,0]
total = tn+tp+fp+fn
print("Kalman Detection Accuracy: ",np.round(100*(tn+tp)/total,2))
print("Kalman Detection Precision: ",np.round(100*tp/(tp+fp)),2)
print("Kalman Detection Recall: ",np.round(100*tp/(tp+fn)),2)

detTime = KalmanData[boolClosedFault]['detTime'].copy()
fTime = KalmanData[boolClosedFault]['fTime'].copy()
detTime[detTime == 0 ] = np.nan
print("Median closed fault detection time",np.nanmedian(detTime-fTime))
detTime = KalmanData[boolOpenFault]['detTime'].copy()
fTime = KalmanData[boolOpenFault]['fTime'].copy()
detTime[detTime == 0 ] = np.nan
print("Median open fault detection time",np.nanmedian(detTime-fTime))


#%% Confusion matrix for closed/open faults
cmClosed = confusion_matrix(np.concatenate(KalmanData[boolClosedFault]['faultVector'].to_numpy())>0,
                            np.concatenate(KalmanData[boolClosedFault]['isoVector'].to_numpy())>0)
tn = cmClosed[0,0]
tp = cmClosed[1,1]
fp = cmClosed[0,1]
fn = cmClosed[1,0]

spec = tn/(tn+fp)
sens = tp/(tp+fn) #sensitivity aka recall
acc= (tp+tn)/(tp+tn+fp+fn)
total = tp+tn+fp+fn

ppv = tp/(fp+tp) # positive predictive value, aka precision
npv = tn/(fn+tn)
beta = 1/2 # How much more important is sensitivity over precisiion
f1 = (1+beta**2)*ppv*sens/((beta**2*ppv)+sens)
print("Closed Fault accuracy: ", 100*acc)
print("Closed Fault precision: ", 100*ppv)
print("Closed Fault recall: ", 100*sens)

cmOpen = confusion_matrix(np.concatenate(KalmanData[boolOpenFault]['faultVector'].to_numpy())>0,
                          np.concatenate(KalmanData[boolOpenFault]['isoVector'].to_numpy())>0)
tn = cmOpen[0,0]
tp = cmOpen[1,1]
fp = cmOpen[0,1]
fn = cmOpen[1,0]

spec = tn/(tn+fp)
sens = tp/(tp+fn) #sensitivity aka recall
acc= (tp+tn)/(tp+tn+fp+fn)
total = tp+tn+fp+fn

ppv = tp/(fp+tp) # positive predictive value, aka precision
npv = tn/(fn+tn)
beta = 1/2 # How much more important is sensitivity over precisiion
f1 = (1+beta**2)*ppv*sens/((beta**2*ppv)+sens)
print("Open Fault accuracy: ",100*acc)
print("Open Fault precision: ", 100*ppv)
print("Open Fault recall: ", 100*sens)
#%%
# Isolation Recall (row normalized confusion matrix)
cmIsoPure = confusion_matrix(faultCombined[isoCombined>0],isoCombined[isoCombined>0])
cmPairPure = confusion_matrix(faultPair[isoPair>0],isoPair[isoPair>0])

cmRow = cmIsoPure.astype('float') / cmIsoPure.sum(axis=1)[:, np.newaxis]
isoRec = np.nanmean(np.diag(cmRow))
cmCol = cmIsoPure.astype('float') / cmIsoPure.sum(axis=0)[np.newaxis,:]
isoPrec = np.nanmean(np.diag(cmCol))
print("Kalman Isolation Accuracy:", np.round(100*acc,2))
print("Kalman Isolation Precision: ", np.round(100*isoPrec,2))
print("Kalman Isolation Recall:", np.round(100*isoRec,2))

# Isolation Recall (row normalized confusion matrix)
cmpRow = cmPairPure.astype('float') / cmPairPure.sum(axis=1)[:, np.newaxis]
isoRec = np.nanmean(np.diag(cmpRow))
print("Kalman Pair Isolation Recall: ", isoRec)  
# Isolation Precision (column normalized confusion matrix)
cmpCol = cmPairPure.astype('float') / cmPairPure.sum(axis=0)[np.newaxis,:]
isoPrec = np.nanmean(np.diag(cmpCol))
print("Kalman Pair Isolation Precision: ", isoPrec)
#%% Plot Comfusion Matrix
detClasses = {0:"Faultless",1:"Faulty"}
# isoclasses = {0:'Faultless', 
#               1: "S1, T1",2: "S1, T2",3: "S1, T3",4: "S1, T4",5: "S1, T5",6: "S1, T6",
#               7: "S2, T1",8: "S2, T2",9: "S2, T3",10:"S2, T4",11:"S2, T5",12:"S2, T6",
#               13:"S3, T1",14:"S3, T2",15:"S3, T3",16:"S3, T4",17:"S3, T5",18:"S3, T6",
#               19:"S4, T1",20:"S4, T2",21:"S4, T3",22:"S4, T4",23:"S4, T5",24:"S4, T6",
#               25:"S5, T1",26:"S5, T2",27:"S5, T3",28:"S5, T4",29:"S5, T5",30:"S5, T6",
#               31:"S6, t1",32:"S6, T2",33:"S6, T3",34:"S6, T4",35:"S6, T5",36:"S6, T6",}
# pairClasses = {0:'Faultless', 
#               1: "S1, TP1",2: "S1, TP2",3: "S1, TP3",4: "S2, TP1",5: "S2, TP2",6: "S2, TP3",
#               7: "S3, TP1",8: "S3, TP2",9: "S3, TP3",10:"S4, TP1",11:"S4, TP2",12:"S4, TP3",
#               13:"S5, TP1",14:"S5, TP2",15:"S5, TP3",16:"S6, TP1",17:"S6, TP2",18:"S6, TP3"}
isoclasses = {0: "S1, T1",1: "S1, T2",2: "S1, T3",3: "S1, T4",4: "S1, T5",5: "S1, T6",
              6: "S2, T1",7: "S2, T2",8: "S2, T3",9: "S2, T4",10:"S2, T5",11:"S2, T6",
              12:"S3, T1",13:"S3, T2",14:"S3, T3",15:"S3, T4",16:"S3, T5",17:"S3, T6",
              18:"S4, T1",19:"S4, T2",20:"S4, T3",21:"S4, T4",22:"S4, T5",23:"S4, T6",
              24:"S5, T1",25:"S5, T2",26:"S5, T3",27:"S5, T4",28:"S5, T5",29:"S5, T6",
              30:"S6, t1",31:"S6, T2",32:"S6, T3",33:"S6, T4",34:"S6, T5",35:"S6, T6",}
pairClasses = {0: "S1, TP1",1: "S1, TP2",2: "S1, TP3",3: "S2, TP1",4: "S2, TP2",5: "S2, TP3",
              6: "S3, TP1",7: "S3, TP2",8: "S3, TP3",9: "S2, TP4",10:"S4, TP2",11:"S4, TP3",
              12:"S5, TP1",13:"S5, TP2",14:"S5, TP3",15:"S6, TP4",16:"S6, TP2",17:"S6, TP3"}
plot_confusion_matrix(cmDet,faultDet,detCombined,detClasses,normalize ="row",title= "Kalman Fault Detection Confusion Matrix")
if save:
    plt.savefig(figurePath + "Kalman"+ fileSuffix +"FaultDetection_ConfusionMatrix.pdf")
plot_confusion_matrix(cmIsoPure,faultCombined[isoCombined>0]-1,isoCombined[isoCombined>0]-1,isoclasses,normalize ="row",title= "Kalman Fault Isolation Confusion Matrix",textsize = 9)
if save:
    plt.savefig(figurePath + "KalmanFaultIsolation_ConfusionMatrix.pdf")
plot_confusion_matrix(cmPairPure,faultPair[isoPair>0]-1,isoPair[isoPair>0]-1,pairClasses,normalize ="row",title= "Kalman Thruster Pair (TP) Isolation Confusion Matrix",textsize = 16)
# if save:
plt.savefig(figurePath + "KalmanFaultPairIsolation_ConfusionMatrix.pdf")

# Determine fault detection delay vs intensity per fault
detDelayClosed = KalmanData[boolClosedFault]['detTime'].copy().to_numpy()-KalmanData[boolClosedFault]['fTime'].copy().to_numpy()
detDelayOpen = KalmanData[boolOpenFault]['detTime'].copy().to_numpy()-KalmanData[boolOpenFault]['fTime'].copy().to_numpy()

paramsClosed = KalmanData[boolClosedFault]['fParam'].copy().to_numpy()
paramsOpen = KalmanData[boolOpenFault]['fParam'].copy().to_numpy()


#%% Determine average reaction time per fault over a 0.1 fault intensity interval
delayClosed = np.zeros([9,1])
delayOpen = np.zeros([9,1])
for i in range(0,9):
    para = 0.1 + i/10
    boolPara = (KalmanData['fParam'] >para) & (KalmanData['fParam'] < para + 0.1)
    tempDelayClosed = KalmanData[boolPara & boolClosedFault]['detTime'].copy().to_numpy()-KalmanData[boolPara & boolClosedFault]['fTime'].copy().to_numpy()
    tempDelayOpen = KalmanData[boolPara & boolOpenFault]['detTime'].copy().to_numpy()-KalmanData[boolPara & boolOpenFault]['fTime'].copy().to_numpy()
    
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
    plt.savefig(figurePath + "KalmanOpen_DetectionTime_BoxPlot.pdf")

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
    plt.savefig(figurePath + "KalmanClosed_DetectionTime_Linear.pdf")

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
    plt.savefig(figurePath + "KalmanOpen_DetectionTime_Linear.pdf")
#%% Compute accuracy versus fault intensity
outputStack = np.stack(KalmanData['isoVector'].to_numpy())
trueStack = np.stack(KalmanData['faultVector'].to_numpy())
accVsParam = np.zeros([outputStack.shape[0],5])
for i in range(outputStack.shape[0]-1):
    accVsParam[i,0] = KalmanData['fParam'].iloc[i]
    conf = confusion_matrix(trueStack[i,:],outputStack[i,:]>0)
    total =np.sum(conf)
    accVsParam[i,1] = conf[0,0]/total
    accVsParam[i,2] = conf[1,1]/total
    accVsParam[i,3] = conf[0,1]/total
    accVsParam[i,4] = conf[1,0]/total
accVsParamDF = pd.DataFrame(accVsParam, columns = ['fParam','tn','tp','fp','fn'])
#%% Gather into boxes of 0.1
gathered = np.zeros([9,4])
paraArray = np.arange(0,9)*0.1 + 0.1
cmParaClosed = []
cmParaOpen = []
for para in paraArray:
    boolPara = (KalmanData['fParam'] >para) & (KalmanData['fParam'] < para + 0.1)
    outPara  = np.concatenate(KalmanData[boolPara & boolClosedFault]['isoVector'].copy().to_numpy())
    truePara = np.concatenate(KalmanData[boolPara & boolClosedFault]['faultVector'].copy().to_numpy())
    cmParaClosed.append(confusion_matrix(truePara > 0,outPara > 0))
    
    outPara  = np.concatenate(KalmanData[boolPara & boolOpenFault]['isoVector'].copy().to_numpy())
    truePara = np.concatenate(KalmanData[boolPara & boolOpenFault]['faultVector'].copy().to_numpy())
    cmParaOpen.append(confusion_matrix(truePara > 0,outPara > 0))

cmParaOpen = np.array(cmParaOpen)
cmParaClosed = np.array(cmParaClosed)
#%% Quality Plots
tnPara = cmParaClosed[:,0,0]
tpPara = cmParaClosed[:,1,1]
fpPara = cmParaClosed[:,0,1]
fnPara = cmParaClosed[:,1,0]

specPara = tnPara/(tnPara+fpPara)
sensPara = tpPara/(tpPara+fnPara) #sensitivity aka recall
accPara = (tpPara+tnPara)/(tpPara+tnPara+fpPara+fnPara)
totalPara = tpPara + tnPara + fpPara + fnPara

ppvPara = tpPara/(fpPara + tpPara) # positive predictive value, aka precision
npvPara = tnPara/(fnPara + tnPara)
f1Para = (1+beta**2)*ppvPara*sensPara/((beta**2*ppvPara)+sensPara)

    
plt.figure()
plt.plot(paraArray,100*tnPara / totalPara,'g',label = 'True Negative')
plt.plot(paraArray,100*fnPara / totalPara,'r',label = 'False Negative')
plt.plot(paraArray,100*tpPara / totalPara,'y',label = 'True Positive')
plt.plot(paraArray,100*fpPara / totalPara,'k',label = 'False Positive')
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.xlabel("Fault Intensity Parameter [-]")
plt.ylim([0,100])
plt.ylabel("Fraction of Dataset[%]")
plt.title("Closed Fault Detection Rates")
plt.legend()
if save:
    plt.savefig(figurePath + "KalmanClosedRates.pdf")

plt.figure()
#plt.plot(paraArray,100*specPara,label = "Specificity")
plt.plot(paraArray,100*sensPara,'r',label = "Recall")
plt.plot(paraArray,100*accPara,'g',label = "Accuracy")
plt.plot(paraArray,100*ppvPara,'b',label = "Precision")
#plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylim([90,101])
plt.ylabel("[%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.legend()
plt.title("Closed Fault Quality Measures")
if save:
    plt.savefig(figurePath +"KalmanClosedQualityMeasures.pdf")

#%% Open
tnPara = cmParaOpen[:,0,0]
tpPara = cmParaOpen[:,1,1]
fpPara = cmParaOpen[:,0,1]
fnPara = cmParaOpen[:,1,0]

specPara = tnPara/(tnPara+fpPara)
sensPara = tpPara/(tpPara+fnPara) #sensitivity aka recall
accPara = (tpPara+tnPara)/(tpPara+tnPara+fpPara+fnPara)
totalPara = tpPara + tnPara + fpPara + fnPara

ppvPara = tpPara/(fpPara + tpPara) # positive predictive value, aka precision
npvPara = tnPara/(fnPara + tnPara)
f1Para = (1+beta**2)*ppvPara*sensPara/((beta**2*ppvPara)+sensPara)


plt.figure()
plt.plot(paraArray,100*tnPara / totalPara,'g',label = 'True Negative')
plt.plot(paraArray,100*fnPara / totalPara,'r',label = 'False Negative')
plt.plot(paraArray,100*tpPara / totalPara,'y',label = 'True Positive')
plt.plot(paraArray,100*fpPara / totalPara,'k',label = 'False Positive')
plt.ylim([0,100])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylabel("Fraction of Dataset[%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.ylabel("Fraction of Dataset")
plt.title("Open Fault Detection Rates")
plt.legend()
if save:
    plt.savefig(figurePath +"KalmanOpenRates.pdf")

plt.figure()
#plt.plot(paraArray,100*specPara,label = "Specificity")
plt.plot(paraArray,100*sensPara,'r',label = "Recall")
plt.plot(paraArray,100*accPara,'g',label = "Accuracy")
plt.plot(paraArray,100*ppvPara,'b',label = "Precision")
#plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
plt.ylim([99.8,100.01])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylabel("[%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.legend()
plt.title("Open Fault Quality Measures")
if save:
    plt.savefig(figurePath +"KalmanOpenQualityMeasures.pdf")
#%%
isoAccVsParam = np.zeros([len(KalmanData),2])
for j in range(len(KalmanData)):
    isoAccVsParam[j,0] = KalmanData['fParam'].iloc[j]
    output = KalmanData['isoVector'].iloc[j]
    true = KalmanData['faultVector'].iloc[j]
    isoCorrect = output[output>0] == true[output>0]
    if len(isoCorrect)>0:
        isoAccVsParam[j,1] = sum(isoCorrect)/len(isoCorrect)
    else:
        isoAccVsParam[j,1] = 0
#%%
isoAccVsParam = pd.DataFrame(isoAccVsParam, columns = ['fParam','isoAcc'])

paraArray = isoAccVsParam['fParam'].copy().to_numpy()
isoAccArray = isoAccVsParam['isoAcc'].copy().to_numpy()

sortIndex = np.argsort(paraArray)
isoAccArray = isoAccArray[sortIndex]
paraArray = paraArray[sortIndex]

nMean = 1
paraArrayMean = running_mean(paraArray,nMean)
isoAccArrayMean = running_mean(isoAccArray,nMean)

# Closed Faults
paraArrayClosed = isoAccVsParam[boolClosedFault]['fParam'].copy().to_numpy()
isoAccArrayClosed = isoAccVsParam[boolClosedFault]['isoAcc'].copy().to_numpy()

sortIndexClosed = np.argsort(paraArrayClosed)
isoAccArrayClosed = isoAccArrayClosed[sortIndexClosed]
paraArrayClosed = paraArrayClosed[sortIndexClosed]


paraArrayMeanClosed = running_mean(paraArrayClosed,nMean)
isoAccArrayMeanClosed = running_mean(isoAccArrayClosed,nMean)
          
#Open Faults
paraArrayOpen = isoAccVsParam[boolOpenFault]['fParam'].to_numpy()
isoAccArrayOpen = isoAccVsParam[boolOpenFault]['isoAcc'].to_numpy()

sortIndexOpen = np.argsort(paraArrayOpen)
isoAccArrayOpen = isoAccArrayOpen[sortIndexOpen]
paraArrayOpen = paraArrayOpen[sortIndexOpen] 

   
paraArrayMeanOpen = running_mean(paraArrayOpen,nMean)
isoAccArrayMeanOpen = running_mean(isoAccArrayOpen,nMean)
plt.figure()
plt.plot(paraArrayMean,100*isoAccArrayMean,label = "Average Isolation Accuracy")
plt.plot(paraArrayMeanClosed,100*isoAccArrayMeanClosed,label = "Closed Fault Isolation Accuracy")
plt.plot(paraArrayMeanOpen  ,100*isoAccArrayMeanOpen  ,label = "Open Fault Isolation Accuracy")
plt.ylim([0,100])
# plt.xscale('log')
# plt.xlim(1e-2,1e-1)
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylabel("Isolation Accuracy [%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.title("Kalman Isolation Accuracy")
plt.legend()
if save:
    plt.savefig(figurePath + "Kalman_IsoAccBoth_FaultIntensity.pdf")
