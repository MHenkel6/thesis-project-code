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
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#%%
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
filePath = "D:/Files/TUDelftLocal/Thesis/Software/Simulation/DataScenario/"
figurePath = "D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\BigFont_figures\\"
scenario = "MultiFail"

save = False
indRun = False
KalmanData = pd.read_hdf(filePath+"CentralizedKalmanCombined_"+ scenario +".hdf")
timeFix = False
if timeFix:        
    scenario2 = scenario + "V2" if not scenario == "NavChangeV2" else "NavChangeV3"
    KalmanData2 = pd.read_hdf(filePath+"CentralizedKalmanCombined_"+ scenario2 +".hdf")
    boolOpenFault = KalmanData['fType'] == 2
    KalmanData = KalmanData[boolOpenFault]
    KalmanData = KalmanData2.append(KalmanData)
print(KalmanData['fSat'].dtype)
print(KalmanData['fThr'].dtype)
print(KalmanData['fType'].dtype)
print(KalmanData['fParam'].dtype)
print(KalmanData['fTime'].dtype)
print(KalmanData['detTime'].dtype)
print(KalmanData['faultVector'].dtype)
print(KalmanData['isoVector'].dtype)
if scenario == "MultiFail":
    print(KalmanData['isoDetailed'].dtype)
#%% Plot Confusion Matrix

boolClosedFault = KalmanData['fType'] == 1
boolOpenFault = KalmanData['fType'] == 2

# Keys of data are ['fSat','fThr','fType','fParam','fTime','detTime','faultVector',isoVector']
if scenario == "MultiFail":
    faultVectorList = []
    for faultVector in KalmanData['faultVector'].to_numpy():
        firstNZ = np.where(faultVector >0 )[0][0]
        correctedVector = np.copy(faultVector)
        correctedVector[firstNZ:firstNZ+100] = 1
        faultVectorList.append(correctedVector)
    faultCombined = np.concatenate(np.array(faultVectorList))
else:
    faultCombined = np.concatenate(KalmanData['faultVector'].to_numpy())
isoCombined = np.concatenate(KalmanData['isoVector'].to_numpy())
detCombined = np.ceil(np.divide(isoCombined,36))
faultDet =  np.ceil(np.divide(faultCombined,36))


isoPair = np.ceil(np.divide(isoCombined,2))
faultPair = np.ceil(np.divide(faultCombined,2))
# Evaluate accuracy 
if scenario == "MultiFail":
    isoCorrect = np.logical_or(isoCombined[isoCombined>0] == faultCombined[isoCombined>0],
                               isoCombined[isoCombined>0] == 1)
else:
    isoCorrect = isoCombined[isoCombined>0] == faultCombined[isoCombined>0]
acc = sum(isoCorrect)/len(isoCorrect)


if scenario == "MultiFail":
    pairIsoCorrect = np.logical_or(isoPair[isoPair>0] == faultPair[isoPair>0],
                                   isoPair[isoPair>0] == 1)
else:
    pairIsoCorrect = isoPair[isoPair>0] == faultPair[isoPair>0]
pairAcc = sum(pairIsoCorrect)/len(pairIsoCorrect)

print("Kalman Pair Isolation Accuracy: ", np.round(100*pairAcc,2))

cmDet = confusion_matrix(faultDet,detCombined)
cmIso = confusion_matrix(faultCombined,isoCombined)
cmPair = confusion_matrix(faultPair,isoPair)
cmIsoPure = confusion_matrix(faultCombined[isoCombined>0],isoCombined[isoCombined>0])
cmPairPure = confusion_matrix(faultPair[isoPair>0],isoPair[isoPair>0])

tn = cmDet[0,0]
tp = cmDet[1,1]
fp = cmDet[0,1]
fn = cmDet[1,0]
total = tn+tp+fp+fn
print("Kalman Detection Accuracy: ",np.round(100*(tn+tp)/total,2))
print("Kalman Detection Precision: ",np.round(100*tp/(tp+fp),2))
print("Kalman Detection Recall: ",np.round(100*tp/(tp+fn),2))
if scenario == "MultiFail":
    cmIsoadapt = np.copy(cmIso)
    cmIsoadapt[np.diag_indices(cmIsoadapt.shape[0])] += cmIsoadapt[:,0]
    cmIsoadapt[0,0] /= 2
    cmIsoadapt[1:,0] = 0 
    cmIso = cmIsoadapt

# Isolation Precision (column normalized confusion matrix)
cmRow = cmIsoPure.astype('float') / cmIsoPure.sum(axis=1)[:, np.newaxis]
isoRec = np.nanmean(np.diag(cmRow))
cmCol = cmIsoPure.astype('float') / cmIsoPure.sum(axis=0)[np.newaxis,:]
isoPrec = np.nanmean(np.diag(cmCol))
print("Kalman Isolation Accuracy:", np.round(100*acc,2))
print("Kalman Isolation Precision: ", np.round(100*isoPrec,2))
print("Kalman Isolation Recall:", np.round(100*isoRec,2))

#%% Plot Comfusion Matrix
detClasses = {0:"Faultless",1:"Faulty"}
isoclasses2 = {0:'Faultless', 
              1: "S1, T1",2: "S1, T2",3: "S1, T3",4: "S1, T4",5: "S1, T5",6: "S1, T6",
              7: "S2, T1",8: "S2, T2",9: "S2, T3",10:"S2, T4",11:"S2, T5",12:"S2, T6",
              13:"S3, T1",14:"S3, T2",15:"S3, T3",16:"S3, T4",17:"S3, T5",18:"S3, T6",
              19:"S4, T1",20:"S4, T2",21:"S4, T3",22:"S4, T4",23:"S4, T5",24:"S4, T6",
              25:"S5, T1",26:"S5, T2",27:"S5, T3",28:"S5, T4",29:"S5, T5",30:"S5, T6",
              31:"S6, t1",32:"S6, T2",33:"S6, T3",34:"S6, T4",35:"S6, T5",36:"S6, T6",}
pairClasses2 = {0:'Faultless', 
              1: "S1, TP1",2: "S1, TP2",3: "S1, TP3",4: "S2, TP1",5: "S2, TP2",6: "S2, TP3",
              7: "S3, TP1",8: "S3, TP2",9: "S3, TP3",10:"S4, TP1",11:"S4, TP2",12:"S4, TP3",
              13:"S5, TP1",14:"S5, TP2",15:"S5, TP3",16:"S6, TP1",17:"S6, TP2",18:"S6, TP3"}
isoclasses = {0: "S1, T1",1: "S1, T2",2: "S1, T3",3: "S1, T4",4: "S1, T5",5: "S1, T6",
              6: "S2, T1",7: "S2, T2",8: "S2, T3",9: "S2, T4",10:"S2, T5",11:"S2, T6",
              12:"S3, T1",13:"S3, T2",14:"S3, T3",15:"S3, T4",16:"S3, T5",17:"S3, T6",
              18:"S4, T1",19:"S4, T2",20:"S4, T3",21:"S4, T4",22:"S4, T5",23:"S4, T6",
              24:"S5, T1",25:"S5, T2",26:"S5, T3",27:"S5, T4",28:"S5, T5",29:"S5, T6",
              30:"S6, t1",31:"S6, T2",32:"S6, T3",33:"S6, T4",34:"S6, T5",35:"S6, T6",}
pairClasses = {0: "S1, TP1",1: "S1, TP2",2: "S1, TP3",3: "S2, TP1",4: "S2, TP2",5: "S2, TP3",
          6: "S3, TP1",7: "S3, TP2",8: "S3, TP3",9: "S2, TP4",10:"S4, TP2",11:"S4, TP3",
          12:"S5, TP1",13:"S5, TP2",14:"S5, TP3",15:"S6, TP4",16:"S6, TP2",17:"S6, TP3"}

    
# plot_confusion_matrix(cmDet,faultDet,detCombined,detClasses,normalize =True,title= "Kalman Fault Detection Confusion Matrix")
# if save:
#     plt.savefig(figurePath + "Kalman"+ scenario +"_FaultDetection_ConfusionMatrix.pdf")
cmIso = confusion_matrix(faultCombined[np.logical_and(isoCombined>0,faultCombined>0)]-1,isoCombined[np.logical_and(isoCombined>0,faultCombined>0)]-1)

plot_confusion_matrix(cmIso,faultCombined[np.logical_and(isoCombined>0,faultCombined>0)]-1,isoCombined[np.logical_and(isoCombined>0,faultCombined>0)]-1,isoclasses,normalize ="row",title= "Kalman Fault Isolation Confusion Matrix",textsize = 12)
if save:
    plt.savefig(figurePath + "Kalman"+ scenario +"_FaultIsolation_ConfusionMatrix.pdf")
cmPair = confusion_matrix(faultPair[np.logical_and(isoPair>0,faultPair>0)]-1,isoPair[np.logical_and(isoPair>0,faultPair>0)]-1)
plot_confusion_matrix(cmPair,faultPair[np.logical_and(isoPair>0,faultPair>0)]-1,isoPair[np.logical_and(isoPair>0,faultPair>0)]-1,pairClasses,normalize ="row",title= "Kalman Thruster Pair (TP) Isolation Confusion Matrix",textsize = 16)
if save:
    plt.savefig(figurePath + "Kalman"+ scenario +"_FaultPairIsolation_ConfusionMatrix.pdf")

# Determine fault detection delay vs intensity per fault
detDelayClosed = KalmanData[boolClosedFault]['detTime'].to_numpy()-KalmanData[boolClosedFault]['fTime'].to_numpy()
detDelayOpen = KalmanData[boolOpenFault]['detTime'].to_numpy()-KalmanData[boolOpenFault]['fTime'].to_numpy()

paramsClosed = KalmanData[boolClosedFault]['fParam'].to_numpy()
paramsOpen = KalmanData[boolOpenFault]['fParam'].to_numpy()

#%% Compute accuracy versus fault intensity
cmParaClosed = []
cmParaOpen = []
outputStack = np.stack(KalmanData['isoVector'].to_numpy())
trueStack = np.stack(KalmanData['faultVector'].to_numpy())
accVsParam = np.zeros([outputStack.shape[0],5])
for i in range(outputStack.shape[0]):
    accVsParam[i,0] = KalmanData['fParam'].iloc[i]
    conf = confusion_matrix(trueStack[i,:]>0,outputStack[i,:]>0)
    total = np.sum(conf)
    accVsParam[i,1] = conf[0,0]/total
    accVsParam[i,2] = conf[1,1]/total
    accVsParam[i,3] = conf[0,1]/total
    accVsParam[i,4] = conf[1,0]/total
accVsParamDF = pd.DataFrame(accVsParam, columns = ['fParam','tn','tp','fp','fn'])

#%% Plot accuracy vs intensity
if scenario == "LowIntensity" or scenario == "LowThrust":
    paraArray = accVsParamDF[boolClosedFault]['fParam'].copy().to_numpy()
    tn = accVsParamDF[boolClosedFault]['tn'].copy().to_numpy()
    tp = accVsParamDF[boolClosedFault]['tp'].copy().to_numpy()
    fp = accVsParamDF[boolClosedFault]['fp'].copy().to_numpy()
    fn = accVsParamDF[boolClosedFault]['fn'].copy().to_numpy()
    
    total = tn+tp+fp+fn
    tnPara = tn / total
    tpPara = tp / total
    fpPara = fp / total
    fnPara = fn / total 
    # Sort arrays
    sortIndex = np.argsort(paraArray)
    tnPara = tnPara[sortIndex]
    tpPara = tpPara[sortIndex]
    fpPara = fpPara[sortIndex]
    fnPara = fnPara[sortIndex]
    paraArray = paraArray[sortIndex]
    specPara = tnPara/(tnPara+fpPara)
    sensPara = tpPara/(tpPara+fnPara) #sensitivity aka recall
    accPara = (tpPara+tnPara)/(tpPara+tnPara+fpPara+fnPara)
    totalPara = tpPara + tnPara + fpPara + fnPara
    
    ppvPara = tpPara/(fpPara + tpPara) # positive predictive value, aka precision
    npvPara = tnPara/(fnPara + tnPara)
    beta = 1/2
    f1Para = (1+beta**2)*ppvPara*sensPara/((beta**2*ppvPara)+sensPara)
    
     # Calculate mean over 5 elements
    nMean = 1
    # tnParaMean = np.array([np.nanmean(tnPara[nMean*x:nMean*x+nMean]) for x in range(len(tnPara)//nMean)])
    # fnParaMean = np.array([np.nanmean(fnPara[nMean*x:nMean*x+nMean]) for x in range(len(fnPara)//nMean)])
    # tpParaMean = np.array([np.nanmean(tpPara[nMean*x:nMean*x+nMean]) for x in range(len(tpPara)//nMean)])
    # fpParaMean = np.array([np.nanmean(fpPara[nMean*x:nMean*x+nMean]) for x in range(len(fpPara)//nMean)])
    
    # sensParaMean  = np.array([np.nanmean(sensPara[nMean*x:nMean*x+nMean]) for x in range(len(sensPara)//nMean)])
    # ppvParaMean   = np.array([np.nanmean(ppvPara[nMean*x:nMean*x+nMean]) for x in range(len(ppvPara)//nMean)])
    # accParaMean   = np.array([np.nanmean(accPara[nMean*x:nMean*x+nMean]) for x in range(len(accPara)//nMean)])
    # paraArrayMean = np.array([np.nanmean(paraArray[nMean*x:nMean*x+nMean]) for x in range(len(paraArray)//nMean)])
    # Moving Average
    tnParaMean = running_mean(tnPara, nMean)
    fnParaMean = running_mean(fnPara, nMean)
    tpParaMean = running_mean(tpPara, nMean)
    fpParaMean = running_mean(fpPara, nMean)
    
    sensParaMean  = running_mean(sensPara, nMean)
    ppvParaMean   = running_mean(ppvPara, nMean)
    accParaMean   = running_mean(accPara, nMean)
    paraArrayMean = running_mean(paraArray, nMean)
    
    plt.figure()
    plt.plot(paraArrayMean,100*tnParaMean,'g',label = 'True Negative')
    plt.plot(paraArrayMean,100*fnParaMean,'r',label = 'False Negative')
    plt.plot(paraArrayMean,100*tpParaMean,'y',label = 'True Positive')
    plt.plot(paraArrayMean,100*fpParaMean,'k',label = 'False Positive')
    plt.xscale('log' if scenario == "LowIntensity" else 'linear')
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.xlabel("Fault Intensity Parameter [-]")
    plt.ylim([0,100])
    plt.ylabel("Fraction of Dataset[%]")
    plt.title("Kalman Closed Fault Detection Rates")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(figurePath +"Kalman"+ scenario +"_ClosedRates.pdf")
    
    plt.figure()
    #plt.plot(paraArray,100*specPara,label = "Specificity")
    plt.plot(paraArrayMean,100*sensParaMean,'r' ,label = "Recall")
    plt.plot(paraArrayMean,100*accParaMean,'g' ,label = "Accuracy")
    plt.plot(paraArray,100*ppvPara,'b' ,label = "Precision")
    #plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
    plt.xscale('log' if scenario == "LowIntensity" else 'linear')
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.ylim([0,102])
    plt.ylabel("[%]")
    plt.xlabel("Fault Intensity Parameter [-]")
    plt.legend()
    plt.title("Kalman Closed Fault Quality Measures")
    plt.tight_layout()
    if save:
        plt.savefig(figurePath + "Kalman"+ scenario +"_ClosedQualityMeasures.pdf")
    
    nMean = 1
    paraArray = accVsParamDF[boolOpenFault]['fParam'].copy().to_numpy()
    tn = accVsParamDF[boolOpenFault]['tn'].copy().to_numpy()
    tp = accVsParamDF[boolOpenFault]['tp'].copy().to_numpy()
    fp = accVsParamDF[boolOpenFault]['fp'].copy().to_numpy()
    fn = accVsParamDF[boolOpenFault]['fn'].copy().to_numpy()
    
    total = tn+tp+fp+fn
    tnPara = tn / total
    tpPara = tp / total
    fpPara = fp / total
    fnPara = fn / total 
    # Sort arrays
    sortIndex = np.argsort(paraArray)
    tnPara = tnPara[sortIndex]
    tpPara = tpPara[sortIndex]
    fpPara = fpPara[sortIndex]
    fnPara = fnPara[sortIndex]
    paraArray = paraArray[sortIndex]
    specPara = tnPara/(tnPara+fpPara)
    sensPara = tpPara/(tpPara+fnPara) #sensitivity aka recall
    accPara = (tpPara+tnPara)/(tpPara+tnPara+fpPara+fnPara)
    totalPara = tpPara + tnPara + fpPara + fnPara
    
    ppvPara = tpPara/(fpPara + tpPara) # positive predictive value, aka precision
    npvPara = tnPara/(fnPara + tnPara)
    f1Para = (1+beta**2)*ppvPara*sensPara/((beta**2*ppvPara)+sensPara)
    
     # Calculate mean over 5 elements
    # tnParaMean = np.array([np.nanmean(tnPara[nMean*x:nMean*x+nMean]) for x in range(len(tnPara)//nMean)])
    # fnParaMean = np.array([np.nanmean(fnPara[nMean*x:nMean*x+nMean]) for x in range(len(fnPara)//nMean)])
    # tpParaMean = np.array([np.nanmean(tpPara[nMean*x:nMean*x+nMean]) for x in range(len(tpPara)//nMean)])
    # fpParaMean = np.array([np.nanmean(fpPara[nMean*x:nMean*x+nMean]) for x in range(len(fpPara)//nMean)])
    
    # sensParaMean  = np.array([np.nanmean(sensPara[nMean*x:nMean*x+nMean]) for x in range(len(sensPara)//nMean)])
    # ppvParaMean   = np.array([np.nanmean(ppvPara[nMean*x:nMean*x+nMean]) for x in range(len(ppvPara)//nMean)])
    # accParaMean   = np.array([np.nanmean(accPara[nMean*x:nMean*x+nMean]) for x in range(len(accPara)//nMean)])
    # paraArrayMean = np.array([np.nanmean(paraArray[nMean*x:nMean*x+nMean]) for x in range(len(paraArray)//nMean)])
    # Moving Average
    tnParaMean = running_mean(tnPara, nMean)
    fnParaMean = running_mean(fnPara, nMean)
    tpParaMean = running_mean(tpPara, nMean)
    fpParaMean = running_mean(fpPara, nMean)
    
    sensParaMean  = running_mean(sensPara, nMean)
    ppvParaMean   = running_mean(ppvPara, nMean)
    accParaMean   = running_mean(accPara, nMean)
    paraArrayMean = running_mean(paraArray, nMean)
    
    plt.figure()
    plt.plot(paraArrayMean,100*tnParaMean,'g',label = 'True Negative')
    plt.plot(paraArrayMean,100*fnParaMean,'r',label = 'False Negative')
    plt.plot(paraArrayMean,100*tpParaMean,'y',label = 'True Positive')
    plt.plot(paraArrayMean,100*fpParaMean,'k',label = 'False Positive')
    plt.ylim([0,100])
    plt.xscale('log' if scenario == "LowIntensity" else 'linear')
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.ylabel("Fraction of Dataset[%]")
    plt.xlabel("Fault Intensity Parameter [-]")
    plt.ylabel("Fraction of Dataset")
    plt.title("Kalman Open Fault Detection Rates")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(figurePath + "Kalman"+ scenario +"_OpenRates.pdf")
    
    plt.figure()
    #plt.plot(paraArray,100*specPara,label = "Specificity")
    plt.plot(paraArrayMean,100*sensParaMean,'r',label = "Recall")
    plt.plot(paraArrayMean,100*accParaMean,'g',label = "Accuracy")
    plt.plot(paraArray,100*ppvPara,'b',label = "Precision")
    #plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
    #plt.ylim([80,100])
    plt.xscale('log' if scenario == "LowIntensity" else 'linear')
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.ylabel("[%]")
    plt.xlabel("Fault Intensity Parameter [-]")
    plt.legend()
    plt.title("Kalman Open Fault Quality Measures")
    plt.tight_layout()
    if save:
        plt.savefig(figurePath + "Kalman"+ scenario +"_OpenQualityMeasures.pdf")
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
    
detTime = KalmanData[boolClosedFault]['detTime'].copy()
fTime = KalmanData[boolClosedFault]['fTime'].copy()
detTime[detTime == 0 ] = np.nan
print("Median closed fault detection time",np.nanmedian(detTime-fTime))
detTime = KalmanData[boolOpenFault]['detTime'].copy()
fTime = KalmanData[boolOpenFault]['fTime'].copy()
detTime[detTime == 0 ] = np.nan
print("Median open fault detection time",np.nanmedian(detTime-fTime))

#%% Determine average reaction time per fault over a 0.1 fault intensity interval
if not scenario == 'LowIntensity':
    boxLabels = ["[0.1,0.2)","[0.2,0.3)","[0.3,0.4)","[0.4,0.5)",
                 "[0.5,0.6)","[0.6,0.7)","[0.7,0.8)","[0.8,0.9)",
                 "[0.9,1.0]"]
    delayClosed = np.zeros([9,1])
    delayOpen = np.zeros([9,1])
    detDelayArrayClosed = []
    detDelayArrayOpen = []
    for i in range(0,9):
        para = 0.1 + i*0.9/10
        boolPara = (KalmanData['fParam'] >para) & (KalmanData['fParam'] < para + 0.1)
        tempDelayClosed = KalmanData[boolPara & boolClosedFault]['detTime'].to_numpy()-KalmanData[boolPara & boolClosedFault]['fTime'].to_numpy()
        tempDelayOpen = KalmanData[boolPara & boolOpenFault]['detTime'].to_numpy()-KalmanData[boolPara & boolOpenFault]['fTime'].to_numpy()
        
        delayClosed[i] = np.nanmean(tempDelayClosed)
        delayOpen[i] = np.nanmean(tempDelayOpen)
        detDelayArrayClosed.append(tempDelayClosed.transpose())
        detDelayArrayOpen.append(tempDelayOpen.transpose())
    #detDelayArrayOpen = np.array(detDelayArrayOpen)
    #detDelayArrayClosed = np.array(detDelayArrayClosed)
    
    detDelayClosed = KalmanData[boolClosedFault]['detTime'].to_numpy()-KalmanData[boolClosedFault]['fTime'].to_numpy()
    detDelayOpen = KalmanData[boolOpenFault]['detTime'].to_numpy()-KalmanData[boolOpenFault]['fTime'].to_numpy()
    
    boxLabelsDict = dict(enumerate(boxLabels))
    
    fig,ax = plt.subplots(figsize=[1.25*6.4, 1.25*4.8])
    detDelayClosed_Plot = pd.DataFrame(detDelayArrayClosed).transpose()
    df = detDelayClosed_Plot.rename(columns = boxLabelsDict)
    sns.boxplot(data=df,showfliers = False,color="royalblue")
    plt.title("Distribution of Closed Fault Detection Time")
    plt.ylabel("Detection Time [s]")
    plt.xlabel("Fault Intensity Interval [-]")
    bottom, top = plt.ylim()  
    #plt.ylim([0,top])
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(figurePath + "KalmanClosed"+ scenario +"_DetectionTime_BoxPlot.pdf")
    
    fig,ax = plt.subplots(figsize=[1.25*6.4, 1.25*4.8])
    detDelayOpen_Plot = pd.DataFrame(detDelayArrayOpen).transpose()
    df = detDelayOpen_Plot.rename(columns = boxLabelsDict)
    sns.boxplot(data=df,showfliers = False,color="royalblue")
    plt.ylabel("Detection Time [s]")
    plt.xlabel("Fault Intensity Interval [-]")
    plt.title("Distribution of Open Fault Detection Time")
    bottom, top = plt.ylim()  
    #plt.ylim([0,top])
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(figurePath + "KalmanOpen"+ scenario +"_DetectionTime_BoxPlot.pdf")
    
    #%% Plot Reaction time vs Intensity 

    delayClosedFine = KalmanData[boolClosedFault]['detTime'].copy().to_numpy()-KalmanData[boolClosedFault]['fTime'].copy().to_numpy()
    delayOpenFine = KalmanData[boolOpenFault]['detTime'].copy().to_numpy()-KalmanData[boolOpenFault]['fTime'].copy().to_numpy()
    paraArrayClosed = KalmanData[boolClosedFault]['fParam'].copy().to_numpy()
    paraArrayOpen = KalmanData[boolOpenFault]['fParam'].copy().to_numpy()
    
    delayClosedFine[delayClosedFine<0] = np.nan
    delayOpenFine[delayOpenFine<0] = np.nan
    
    sortIndexClosed = np.argsort(paraArrayClosed)
    delayClosedFine = delayClosedFine[sortIndexClosed]
    paraArrayClosed = paraArrayClosed[sortIndexClosed]
    
    sortIndexOpen = np.argsort(paraArrayOpen)
    delayOpenFine = delayOpenFine[sortIndexOpen]
    paraArrayOpen = paraArrayOpen[sortIndexOpen]
    if not scenario == "MultiFail":
        # closedTrend = np.polyfit(paraArray,delayClosedFine,1)
        
        plt.figure()
        plt.plot(paraArrayClosed,delayClosedFine,label='Closed Fault Detection Time')
        plt.title("Kalman Fault Detection Time")
        plt.plot(paraArrayOpen,delayOpenFine,label='Open Fault Detection Time')
        plt.xlim([np.min(paraArrayOpen),np.max(paraArrayOpen)])
        plt.xscale('log' if scenario == "LowIntensity" else 'linear')
        plt.ylabel("Detection Time [s]")
        plt.xlabel("Fault Intensity [-]")
        plt.minorticks_on()
        plt.grid(b = True, which = 'both')
        plt.legend()
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig(figurePath + "KalmanBoth"+ scenario +"_DetectionTime_Linear.pdf")    
#%%
if scenario == "LowThrust" or scenario == "LowIntensity":
    isoAccVsParam = np.zeros([len(KalmanData),2])
    for j in range(len(KalmanData)):
        isoAccVsParam[j,0] = KalmanData['fParam'].copy().iloc[j]
        output = KalmanData['isoVector'].copy().iloc[j]
        true = KalmanData['faultVector'].copy().iloc[j]
        isoCorrect = output[output>0] == true[output>0]
        if len(isoCorrect)>0:
            isoAccVsParam[j,1] = sum(isoCorrect)/len(isoCorrect)
        else:
            isoAccVsParam[j,1] = 0
    
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
    paraArrayOpen = isoAccVsParam[boolOpenFault]['fParam'].copy().to_numpy()
    isoAccArrayOpen = isoAccVsParam[boolOpenFault]['isoAcc'].copy().to_numpy()
    
    sortIndexOpen = np.argsort(paraArrayOpen)
    isoAccArrayOpen = isoAccArrayOpen[sortIndexOpen]
    paraArrayOpen = paraArrayOpen[sortIndexOpen] 
    
       
    paraArrayMeanOpen = running_mean(paraArrayOpen,nMean)
    isoAccArrayMeanOpen = running_mean(isoAccArrayOpen,nMean)
    plt.figure()
    # plt.plot(paraArrayMean,100*isoAccArrayMean,label = "Average Isolation Accuracy")
    plt.plot(paraArrayMeanClosed,100*isoAccArrayMeanClosed,label = "Closed Fault Isolation Accuracy")
    plt.plot(paraArrayMeanOpen  ,100*isoAccArrayMeanOpen  ,label = "Open Fault Isolation Accuracy")
    plt.ylim([-1,102])
    plt.xscale('log' if scenario == "LowIntensity" else 'linear')
    # plt.xlim(1e-2,1e-1)
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.ylabel("Isolation Accuracy [%]")
    plt.xlabel("Fault Intensity Parameter [-]")
    plt.title("Kalman Isolation Accuracy")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(figurePath + "Kalman"+ scenario +"_IsoAccBoth_FaultIntensity.pdf")

# %%Plot Time Reponse
if indRun:
    sat = 1
    thr = 1
    fty = 2
    minPara = 0.015
    maxPara = 0.9
    boolSat = KalmanData['fSat'] == sat
    boolThr = KalmanData['fThr'] == thr
    boolF = KalmanData['fType'] == fty
    boolParaLOW = (KalmanData['fParam'] >= minPara) 
    boolParaUP =  (KalmanData['fParam']<maxPara)
    boolSelect = boolSat & boolThr & boolF & boolParaLOW & boolParaUP
    
    run = KalmanData[boolSelect].index[0]
    true = KalmanData.iloc[run]['faultVector']>0
    fault = KalmanData.iloc[run]['isoVector']>0
    
    faultType = KalmanData.iloc[run]['fType']
    fSat = KalmanData.iloc[run]['fSat']
    fThr = KalmanData.iloc[run]['fThr']
    fParam = KalmanData.iloc[run]['fParam']
    fTime = KalmanData.iloc[run]['fTime']
    detTime = KalmanData.iloc[run]['detTime']
    
    # fileNameRun = KalmanData.iloc[run]['fName']
    if faultType ==1:
        faultString = "Closed Fault" if faultType == 1 else "Open Fault"
        titleString = faultString + " in Satellite " + str(fSat) + " thruster "+str(fThr) + ", intensity {:.2f}".format(fParam) 
    elif faultType == 2:
        faultString = "Closed Fault" if faultType == 1 else "Open Fault"
        titleString = faultString + " in Satellite " + str(fSat) + " thruster "+str(fThr) + ", intensity {:.2f}".format(fParam) 
    else:
        faultString = "Faultless"
        titleString = "Faultless Case"
    
    plt.figure()
    plt.plot(true,'b',label = 'True Fault Signal')
    plt.plot(fault,'r',label = 'Kalman Output')
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.xlabel('time [s]')
    plt.ylabel('Fault Signal')
    plt.title("Kalman " + titleString)
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(figurePath + "Kalman_" + scenario + "_" + faultString.replace(" ", "")+"_"+ str(run) +".pdf")
#%%
if scenario == "MultiFail":
    accSecondary = []
    pairAccSecondary = []

    ratioListNormal = []
    ratioList = []
    for j in range(KalmanData.shape[0]):
        kalOutput = np.transpose(np.reshape(KalmanData['isoDetailed'].iloc[j],[36,5902]))
        topChoicesList = []
        fTime = np.int(np.ceil(KalmanData['fTime'].iloc[j]))
        kalOutput = kalOutput[np.ceil(fTime).astype(int):,:]
        for i,kalRow in enumerate(kalOutput):
            topChoices = np.argsort(-kalRow)[:2] # Select top two indices
            ratio = kalRow[np.argsort(-kalRow)[0]]/kalRow[np.argsort(-kalRow)[1]]
            topChoicesList.append(topChoices)
            if i<100:
                ratioListNormal.append(ratio)
            else:
                ratioList.append(ratio)
        topChoicesList = np.array(topChoicesList)
        trueVec = KalmanData['faultVector'].iloc[j]
        accSecondary.append((np.sum(topChoicesList[fTime:,:] == trueVec[fTime]),len(topChoicesList[:,0])))
        pairAccSecondary.append((np.sum(np.ceil(topChoicesList[fTime:,:]/2) == np.ceil(trueVec[fTime]/2)),len(topChoicesList[:,0])))
 
    ratioList = np.array(ratioList)
    ratioListNormal = np.array(ratioListNormal)
    ratioPlot = pd.DataFrame([ratioListNormal,ratioList]).transpose()
    df = ratioPlot.rename(columns = {0:"Ratio in single-fault case",1:"Ratio in multi-fault scenario"})
    
    #%%    
    plt.figure(figsize=[1.4*6.4, 1.0*4.8])
    sns.boxplot(data=df,showfliers = False,color="royalblue")
    plt.title("Kalman filter ratio of largest component to second largest")
    plt.ylabel("Ratio between likelihood tests [-]")
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.tight_layout()
    if save:
        plt.savefig(figurePath+"Kalman_" +scenario + "RatioBoxPlot.pdf")
    
    accSecondary = np.array(accSecondary)
    print("Top two accuracy: ",np.sum(accSecondary[:,0])/np.sum(accSecondary[:,1]))
    #%% Select highest 3 components and plot their 
    run = 20
    kalOutput = np.transpose(np.reshape(KalmanData['isoDetailed'].iloc[run],[36,5902]))
    fTime = np.int(np.ceil(KalmanData['fTime'].iloc[run]))
    kalOutput = kalOutput[fTime:,:]
    topChoicesList = []
    ratioList = []
    for kalRow in kalOutput:
        topChoices = np.argsort(-kalRow)[:3] # Select top three indices
        ratio = kalRow[np.argsort(-kalRow)[0]]/kalRow[np.argsort(-kalRow)[1]]
        topChoicesList.append(topChoices)
        ratioList.append(ratio)
    topChoicesList = np.array(topChoicesList)
    ratioList= np.array(ratioList)
    trueVec = KalmanData['faultVector'].iloc[run][fTime:]
    plt.figure()
    plt.plot(np.zeros_like(trueVec),'*b')
    plt.plot(trueVec,'*b')
    plt.plot(topChoicesList[:,0],'.',color ='red')
    plt.plot(topChoicesList[:,1],'.',color ='orange')
    plt.plot(topChoicesList[:,2],'.',color ='yellow')
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.figure()
    plt.plot(ratioList)
    plt.minorticks_on()
    plt.grid(b = True, which = 'both')
    plt.tight_layout()
