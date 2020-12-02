# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 08:31:53 2020

@author: Martin
"""
#%% Import ands Dependencies
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
#%% Function definitions
def detParser(fileName):
    data = []
    with open(fileName,'r') as f:
        counter = 0
        for line in f:
            fTest = line[:4]
            if fTest == "Test" and counter%3 == 0:
                fName = line[:34]
                fSat = int(line[35:36])
                fThr = int(line[37:38])
                fType = int(line[39:40])
                fParam = float(line[41:])
                counter += 1
            elif not fTest == "Test" and counter%3 == 1:
                dataOutput = np.array(line.split(' ')).astype(float)
                counter += 1
            elif not fTest == "Test" and counter%3 == 2:
                dataTrue = np.array(line.split(' ')).astype(float)
                counter += 1
                data.append((fName,fSat,fThr,fType,fParam,dataOutput,dataTrue))
    return data
def nfParser(fileName):
    data = []
    with open(fileName,'r') as f:
        counter = 0
        for line in f:
            fTest = line[:4]
            if fTest == "Test" and counter%2 == 0:
                fName = line[:32]
                fSat = -1
                fThr = -1
                fType = 0
                fParam = 0
                counter += 1
            elif not fTest == "Test" and counter%2 == 1:
                dataOutput = np.array(line.split(' ')).astype(float)
                dataTrue = np.zeros_like(dataOutput)
                counter += 1
                data.append((fName,fSat,fThr,fType,fParam,dataOutput,dataTrue))
    return data
def plot_confusion_matrix(cm,y_true,y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, 
                         annotate=True):
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
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    return ax
def running_mean(x, N):
    cumsum = np.nancumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#%% Load Data
# Select file to read

path=  'D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\Data\\'
figurePath = "D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\BigFont_figures\\"
save = False
locCheck = False
network = 0
fileName = "EvalDetectInd0Corrected.csv" #"EvalDetectNaiveCorrected.csv" #
fileNameNF = fileName[:-4] + "NF.csv"
if fileName == "EvalDetectNaiveCorrected.csv":
    networkName = "Naive Network"
elif fileName =="EvalDetectInd0Corrected.csv" :
    networkName =  "Satellite 1"
else:
    networkName = None
createThresholdCM = False

colNames = ['fName','fSat','fThr','fType','fParam','outputVector','trueVector']
# Read data
dataRaw = detParser(path + fileName)
dataNF = nfParser(path + fileNameNF)
Data = pd.DataFrame(dataRaw + dataNF, columns = colNames)
#Check Data for consistency
dataFault = ~(Data['fType']==0)

fTypeFromName = np.array( [int(fName[10]) for fName in Data[dataFault]['fName'].to_numpy()])
fSatFromName = np.array( [int(fName[12]) for fName in Data[dataFault]['fName'].to_numpy()])
fThrFromName = np.array( [int(fName[13]) for fName in Data[dataFault]['fName'].to_numpy()])

fType = Data[dataFault]['fType'].to_numpy()
fSat  = Data[dataFault]['fSat'].to_numpy()
fThr  = Data[dataFault]['fThr'].to_numpy()
if np.all(fTypeFromName == fType) and np.all(fSatFromName == fSat) and np.all(fThrFromName == fThrFromName):
    print("Data is consistent")

if fileName == "EvalDetectNaiveCorrected.csv":
    outputStackRaw = np.stack(Data['outputVector'].to_numpy())
    
    outputStack = np.zeros([outputStackRaw.shape[0]*6,5552])
    outputStack[0::6,:] =  outputStackRaw[:,0*5552:1*5552]
    outputStack[1::6,:] =  outputStackRaw[:,1*5552:2*5552]
    outputStack[2::6,:] =  outputStackRaw[:,2*5552:3*5552]
    outputStack[3::6,:] =  outputStackRaw[:,3*5552:4*5552]
    outputStack[4::6,:] =  outputStackRaw[:,4*5552:5*5552]
    outputStack[5::6,:] =  outputStackRaw[:,5*5552:6*5552]
    trueStackRaw = np.stack(Data['trueVector'].to_numpy())
    trueStack = np.zeros([trueStackRaw.shape[0]*6,5552]).astype(bool)
    trueStack[0::6,:] =  trueStackRaw[:,0*5552:1*5552]
    trueStack[1::6,:] =  trueStackRaw[:,1*5552:2*5552]
    trueStack[2::6,:] =  trueStackRaw[:,2*5552:3*5552]
    trueStack[3::6,:] =  trueStackRaw[:,3*5552:4*5552]
    trueStack[4::6,:] =  trueStackRaw[:,4*5552:5*5552]
    trueStack[5::6,:] =  trueStackRaw[:,5*5552:6*5552]
    Data = Data.loc[Data.index.repeat(6)].reset_index(drop=True)
    Data['outputVector'] = outputStack.tolist()
    Data['trueVector'] = trueStack.tolist()
else:
    outputStack = np.stack(Data['outputVector'].to_numpy())
    trueStack = np.stack(Data['trueVector'].to_numpy())

boolNoFault = Data['fType'] == 0
boolClosedFault = Data['fType'] == 1
boolOpenFault = Data['fType'] == 2
#%% Determine Threshold
trueCombined = np.concatenate(Data['trueVector'].to_numpy())
outputCombined = np.concatenate(Data['outputVector'].to_numpy())

step = 0.01
detThreshold = np.arange(step,1.0,step)
cmList = [] 
if createThresholdCM:
    tS = time.perf_counter()
    durAverage = 0
    for t in detThreshold:
        cmList.append(confusion_matrix(trueCombined,outputCombined > t))
        tE = time.perf_counter()
        dur = tE-tS
        if durAverage == 0 :
            durAverage = dur
        else:
            durAverage = 0.9*durAverage +0.1*dur 
        tS = tE
        print("Average seconds per run: {}".format(durAverage))
        print("Estimated minutes left: {}".format(100*(1-t)*durAverage/60))
    cmList = np.array(cmList)
    np.save(path + fileName[:-4] + "_ConfusionMatrixThreshold.npy",cmList)
else:
    cmList = np.load(path + fileName[:-4] + "_ConfusionMatrixThreshold.npy")
# Class Definitions 
detClasses = {0:"Faultless",1:"Faulty"}

#%%
tn = cmList[:,0,0]
tp = cmList[:,1,1]
fp = cmList[:,0,1]
fn = cmList[:,1,0]

spec = tn/(tn+fp)
sens = tp/(tp+fn) #sensitivity aka recall
acc= (tp+tn)/(tp+tn+fp+fn)
total = tp+tn+fp+fn

ppv = tp/(fp+tp) # positive predictive value, aka precision
npv = tn/(fn+tn)
beta = 1/2 # How much more important is sensitivity over precisiion
f1 = (1+beta**2)*ppv*sens/((beta**2*ppv)+sens)
fmax = np.argmax(f1)
threshold = detThreshold[fmax]
    
plt.figure()
plt.plot(detThreshold,tn/total,'g',label = 'True Negative')
plt.plot(detThreshold,fn/total,'r',label = 'False Negative')
plt.plot(detThreshold,tp/total,'y',label = 'True Positive')
plt.plot(detThreshold,fp/total,'k',label = 'False Positive')
plt.xlabel("Detection Threshold")
plt.ylabel("Fraction of Dataset")
plt.legend()
plt.figure()
plt.tight_layout()

#plt.plot(detThreshold,spec,label = "Specificity")
plt.plot(detThreshold,sens,label = "Recall")
plt.plot(detThreshold,acc,label = "Accuracy")
plt.plot(detThreshold,ppv,label = "Precision")
#plt.plot(detThreshold,npv,label = "Negative Predictive Value")
plt.plot(detThreshold,f1,label = "F0.5")
plt.plot(threshold,np.max(f1),'k*')
plt.xlabel("Detection Threshold")
plt.ylabel("F Measure")
plt.legend()
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.tight_layout()
plt.show()
if save:
    plt.savefig(figurePath  + networkName.replace(" ", "")+"_DetThreshold_QualityRates.pdf")

print("Maximum F Measure threshold:", threshold)
print("Accuracy:", np.round(100*acc[fmax],2))
print("Precision:",np.round(100*ppv[fmax],2))
print("Recall:",np.round(100*sens[fmax],2))
#%% Confusion Matrix per fault
cmClosed = confusion_matrix(np.concatenate(Data[boolClosedFault]['trueVector'].to_numpy()),
                            np.concatenate(Data[boolClosedFault]['outputVector'].to_numpy())>threshold)
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
print("Closed Fault accuracy: ", np.round(100*acc,2))
print("Closed Fault precision: ", np.round(100*ppv,2))
print("Closed Fault recall: ", np.round(100*sens,2))

cmOpen = confusion_matrix(np.concatenate(Data[boolOpenFault]['trueVector'].to_numpy()),
                          np.concatenate(Data[boolOpenFault]['outputVector'].to_numpy())>threshold)
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
print("Open Fault accuracy: ",np.round(100*acc),2)
print("Open Fault precision: ", np.round(100*ppv,2))
print("Open Fault recall: ", np.round(100*sens,2))
#%% Determine Reaction time
detection = np.ones((5+1,))
detection[0] = 0 
detTime = np.empty(outputStack.shape[0])
detTime[:] = np.nan
fTime = np.zeros(outputStack.shape[0])
faultAssigned = False
detAssigned = False
for i in range(outputStack.shape[0]):
    j = 0
    faultAssigned = False
    detAssigned = False
    while j < outputStack.shape[1]-len(detection):

        det = np.all( (outputStack[i,j:j+len(detection)]>threshold) == detection)
        if det and not detAssigned :
            detTime[i] = j+len(detection)
            detAssigned = True
        fault = np.all(trueStack[i,j:j+len(detection)] == detection)
        if fault and not faultAssigned:
            fTime[i] = j
            faultAssigned = True
        j+=1
        if faultAssigned and detAssigned:
            break

Data['detTime'] = detTime
Data['fTime'] = fTime
#%%
detTimeClosed = Data[boolClosedFault]['detTime'].copy()
fTimeClosed = Data[boolClosedFault]['fTime'].copy()
detTimeClosed[detTimeClosed == 0 ] = np.nan
print("Median closed fault detection time",np.nanmedian(detTimeClosed-fTimeClosed))
detTimeOpen = Data[boolOpenFault]['detTime'].copy()
fTimeOpen = Data[boolOpenFault]['fTime'].copy()
detTimeOpen[detTimeOpen == 0 ] = np.nan
print("Median open fault detection time",np.nanmedian(detTimeOpen-fTimeOpen))

#%% Determine average reaction time per fault over a 0.1 fault intensity interval
delayClosed = np.zeros([9,1])
delayOpen = np.zeros([9,1])
detDelayArrayClosed = []
detDelayArrayOpen = []
for i in range(0,9):
    para = 0.1 + i*0.9/10
    boolPara = (Data['fParam'] >para) & (Data['fParam'] < para + 0.1)
    tempDelayClosed = Data[boolPara & boolClosedFault]['detTime'].to_numpy()-Data[boolPara & boolClosedFault]['fTime'].to_numpy()
    tempDelayOpen = Data[boolPara & boolOpenFault]['detTime'].to_numpy()-Data[boolPara & boolOpenFault]['fTime'].to_numpy()
    
    delayClosed[i] = np.nanmean(tempDelayClosed)
    delayOpen[i] = np.nanmean(tempDelayOpen)
    detDelayArrayClosed.append(tempDelayClosed.transpose())
    detDelayArrayOpen.append(tempDelayOpen.transpose())
#detDelayArrayOpen = np.array(detDelayArrayOpen)
#detDelayArrayClosed = np.array(detDelayArrayClosed)

detDelayClosed = Data[boolClosedFault]['detTime'].to_numpy()-Data[boolClosedFault]['fTime'].to_numpy()
detDelayOpen = Data[boolOpenFault]['detTime'].to_numpy()-Data[boolOpenFault]['fTime'].to_numpy()


boxLabels = ["[0.1,0.2)","[0.2,0.3)","[0.3,0.4)","[0.4,0.5)",
             "[0.5,0.6)","[0.6,0.7)","[0.7,0.8)","[0.8,0.9)",
             "[0.9,1.0]"]
boxLabelsDict = dict(enumerate(boxLabels))

fig,ax = plt.subplots(figsize=[1.25*6.4, 1.25*4.8])
detDelayClosed_Plot = pd.DataFrame(detDelayArrayClosed).transpose()
dfClosed = detDelayClosed_Plot.rename(columns = boxLabelsDict)
sns.boxplot(data=dfClosed,showfliers = False,color="royalblue")
plt.title("Distribution of Closed Fault Detection Time")
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity Interval [-]")
closedbottom, closedtop = plt.ylim()  
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
plt.tight_layout()
plt.show()

if save:
    plt.savefig(figurePath +networkName.replace(" ", "")+"Closed_DetectionTime_BoxPlot.pdf")

fig,ax = plt.subplots(figsize=[1.25*6.4, 1.25*4.8])
detDelayOpen_Plot = pd.DataFrame(detDelayArrayOpen).transpose()
dfOpen = detDelayOpen_Plot.rename(columns = boxLabelsDict)
sns.boxplot(data=dfOpen,showfliers = False,color="royalblue")
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity Interval [-]")
plt.title("Distribution of Open Fault Detection Time")
openbottom, opentop = plt.ylim()  
#plt.ylim([0,top])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
plt.tight_layout()
if save:
    plt.savefig(figurePath  + networkName.replace(" ", "")+"Open_DetectionTime_BoxPlot.pdf")
# Sanity check figure
plt.figure();
plt.plot(Data[boolClosedFault]['fParam'].to_numpy(), Data[boolClosedFault]['detTime'].to_numpy()-Data[boolClosedFault]['fTime'].to_numpy() , '.')
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity [-]")
plt.title("Distribution of Closed Fault Detection Time")
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
plt.show()
if save:
    plt.savefig(figurePath  + networkName.replace(" ", "")+"ScatterPlotDetTimeClosed.pdf")

#%% Plot Reaction time vs Intensity 
steps = 100
delayClosedFine = np.zeros([steps,1])
delayOpenFine = np.zeros([steps,1])
paraArray = np.linspace(0.1,1,steps)
for i in range(0,steps):
    para = 0.1 + i*0.9/steps
    boolPara = (Data['fParam'] > para - 0.00001) & (Data['fParam'] < para + 0.9/steps + 0.00001)
    tempDelayClosed = Data[boolPara & boolClosedFault]['detTime'].to_numpy()-Data[boolPara & boolClosedFault]['fTime'].to_numpy()
    tempDelayOpen = Data[boolPara & boolOpenFault]['detTime'].to_numpy()-Data[boolPara & boolOpenFault]['fTime'].to_numpy()
    delayClosedFine[i] = np.nanmedian(tempDelayClosed)
    delayOpenFine[i] = np.nanmedian(tempDelayOpen)
closedTrend = np.polyfit(paraArray,delayClosedFine,1)
openTrend = np.polyfit(paraArray,delayOpenFine,1)

plt.figure()
plt.plot(paraArray,delayClosedFine,'b',label='Median Detection Time')
plt.plot(paraArray,closedTrend[1]+closedTrend[0]*paraArray,'r',label='Trend line')
plt.ylim([closedbottom,closedtop])
plt.title("Median Closed Fault Detection Time")
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity [-]")
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.legend()
plt.show()
if save:
    plt.savefig(figurePath  + networkName.replace(" ", "")+"Closed_DetectionTime_Linear.pdf")

plt.figure()
plt.plot(paraArray,delayOpenFine,'b',label='Median Detection Time')
plt.plot(paraArray,openTrend[1]+openTrend[0]*paraArray,'r',label='Trend line')
plt.title("Median Open Fault Detection Time")
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity [-]")
plt.ylim([openbottom,opentop])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.legend()
plt.tight_layout()
if save:
    plt.savefig(figurePath + networkName.replace(" ", "")+"Open_DetectionTime_Linear.pdf")

#%% Plot Comfusion Matrix

#plot_confusion_matrix(cmDet,trueCombined,outputCombined,detClasses,normalize =True,title = networkName + " Detection Confusion Matrix")
if save:
    plt.savefig(figurePath + networkName.replace(" ", "")+"FaultDetection_ConfusionMatrix.pdf")

paramsClosed = Data[boolClosedFault]['fParam'].to_numpy()
paramsOpen = Data[boolOpenFault]['fParam'].to_numpy()
#%% Compute accuracy versus fault intensity
cmParaClosed = []
cmParaOpen = []
accVsParam = np.zeros([outputStack[~boolNoFault].shape[0],6])
for i in range(outputStack[~boolNoFault].shape[0]-1):
    accVsParam[i,0] = Data[~boolNoFault]['fParam'].iloc[i]
    accVsParam[i,1] = Data[~boolNoFault]['fType'].iloc[i]
    conf = confusion_matrix(trueStack[i,:],outputStack[i,:]>threshold)
    total =np.sum(conf)
    accVsParam[i,2] = conf[0,0]/total
    accVsParam[i,3] = conf[1,1]/total
    accVsParam[i,4] = conf[0,1]/total
    accVsParam[i,5] = conf[1,0]/total
accVsParamDF = pd.DataFrame(accVsParam, columns = ['fParam','fType','tn','tp','fp','fn'])
boolClosedFaultacc = accVsParamDF['fType'] == 1
boolOpenFaultacc = accVsParamDF['fType'] == 2

#%% Gather into boxes of 0.1
gathered = np.zeros([9,4])
paraArray = np.arange(0,9)*0.1 + 0.1
for para in paraArray:
    boolPara = (Data['fParam'] >para) & (Data['fParam'] < para + 0.1)
    outPara  = np.concatenate(Data[boolPara & boolClosedFaultacc]['outputVector'].to_numpy())
    truePara = np.concatenate(Data[boolPara & boolClosedFaultacc]['trueVector'].to_numpy())
    cmParaClosed.append(confusion_matrix(truePara,outPara > threshold))
    
    outPara  = np.concatenate(Data[boolPara & boolOpenFaultacc]['outputVector'].to_numpy())
    truePara = np.concatenate(Data[boolPara & boolOpenFaultacc]['trueVector'].to_numpy())
    cmParaOpen.append(confusion_matrix(truePara,outPara > threshold))

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
plt.tight_layout()
if save:
    plt.savefig(figurePath +networkName.replace(" ", "")+"ClosedRates.pdf")

plt.figure()
#plt.plot(paraArray,100*specPara,label = "Specificity")
plt.plot(paraArray,100*sensPara,'r',label = "Recall")
plt.plot(paraArray,100*accPara,'g',label = "Accuracy")
plt.plot(paraArray,100*ppvPara,'b',label = "Precision")
#plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylim([0,100])
plt.ylabel("[%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.legend()
plt.title("Closed Fault Quality Measures")
plt.tight_layout()
if save:
    plt.savefig(figurePath +networkName.replace(" ", "")+"ClosedQualityMeasures.pdf")


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
plt.tight_layout()
if save:
    plt.savefig(figurePath +networkName.replace(" ", "")+"OpenRates.pdf")

plt.figure()
#plt.plot(paraArray,100*specPara,label = "Specificity")
plt.plot(paraArray,100*sensPara,'r',label = "Recall")
plt.plot(paraArray,100*accPara,'g',label = "Accuracy")
plt.plot(paraArray,100*ppvPara,'b',label = "Precision")
#plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
if fileName == "EvalDetectInd0Corrected.csv" :
    plt.ylim([99,100])
else:
    plt.ylim([0,100])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylabel("[%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.legend()
plt.title("Open Fault Quality Measures")
plt.tight_layout()
if save:
    plt.savefig(figurePath +networkName.replace(" ", "")+"OpenQualityMeasures.pdf")
#%% Smoother Plots
nMean = 36
paraArray = accVsParamDF[boolClosedFaultacc]['fParam'].to_numpy()
tn = accVsParamDF[boolClosedFaultacc]['tn'].to_numpy()
tp = accVsParamDF[boolClosedFaultacc]['tp'].to_numpy()
fp = accVsParamDF[boolClosedFaultacc]['fp'].to_numpy()
fn = accVsParamDF[boolClosedFaultacc]['fn'].to_numpy()

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
tnParaMean = np.array([np.nanmean(tnPara[nMean*x:nMean*x+nMean]) for x in range(len(tnPara)//nMean)])
fnParaMean = np.array([np.nanmean(fnPara[nMean*x:nMean*x+nMean]) for x in range(len(fnPara)//nMean)])
tpParaMean = np.array([np.nanmean(tpPara[nMean*x:nMean*x+nMean]) for x in range(len(tpPara)//nMean)])
fpParaMean = np.array([np.nanmean(fpPara[nMean*x:nMean*x+nMean]) for x in range(len(fpPara)//nMean)])

sensParaMean  = np.array([np.nanmean(sensPara[nMean*x:nMean*x+nMean]) for x in range(len(sensPara)//nMean)])
ppvParaMean   = np.array([np.nanmean(ppvPara[nMean*x:nMean*x+nMean]) for x in range(len(ppvPara)//nMean)])
accParaMean   = np.array([np.nanmean(accPara[nMean*x:nMean*x+nMean]) for x in range(len(accPara)//nMean)])
paraArrayMean = np.array([np.nanmean(paraArray[nMean*x:nMean*x+nMean]) for x in range(len(paraArray)//nMean)])
# Moving window Average
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

plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.xlabel("Fault Intensity Parameter [-]")
plt.ylim([0,100])
plt.ylabel("Fraction of Dataset[%]")
plt.title(networkName + " Closed Fault Detection Rates")
plt.legend()
plt.tight_layout()
if save:
    plt.savefig(figurePath + networkName.replace(" ", "")+"_SmoothClosedRates.pdf")

plt.figure()
#plt.plot(paraArray,100*specPara,label = "Specificity")
plt.plot(paraArrayMean,100*sensParaMean,'r' ,label = "Recall")
plt.plot(paraArrayMean,100*accParaMean,'g' ,label = "Accuracy")
plt.plot(paraArrayMean,100*ppvParaMean,'b' ,label = "Precision")
#plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylim([0,100])
plt.ylabel("[%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.legend()
plt.title(networkName + " Closed Fault Quality Measures")
plt.tight_layout()
if save:
    plt.savefig(figurePath + networkName.replace(" ", "")+"_SmoothClosedQualityMeasures.pdf")


paraArray = accVsParamDF[boolOpenFaultacc]['fParam'].to_numpy()
tn = accVsParamDF[boolOpenFaultacc]['tn'].to_numpy()
tp = accVsParamDF[boolOpenFaultacc]['tp'].to_numpy()
fp = accVsParamDF[boolOpenFaultacc]['fp'].to_numpy()
fn = accVsParamDF[boolOpenFaultacc]['fn'].to_numpy()

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
tnParaMean = np.array([np.nanmean(tnPara[nMean*x:nMean*x+nMean]) for x in range(len(tnPara)//nMean)])
fnParaMean = np.array([np.nanmean(fnPara[nMean*x:nMean*x+nMean]) for x in range(len(fnPara)//nMean)])
tpParaMean = np.array([np.nanmean(tpPara[nMean*x:nMean*x+nMean]) for x in range(len(tpPara)//nMean)])
fpParaMean = np.array([np.nanmean(fpPara[nMean*x:nMean*x+nMean]) for x in range(len(fpPara)//nMean)])

sensParaMean  = np.array([np.nanmean(sensPara[nMean*x:nMean*x+nMean]) for x in range(len(sensPara)//nMean)])
ppvParaMean   = np.array([np.nanmean(ppvPara[nMean*x:nMean*x+nMean]) for x in range(len(ppvPara)//nMean)])
accParaMean   = np.array([np.nanmean(accPara[nMean*x:nMean*x+nMean]) for x in range(len(accPara)//nMean)])
paraArrayMean = np.array([np.nanmean(paraArray[nMean*x:nMean*x+nMean]) for x in range(len(paraArray)//nMean)])
# Moving window Average
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
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylabel("Fraction of Dataset[%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.ylabel("Fraction of Dataset")
plt.title(networkName + " Open Fault Detection Rates")
plt.legend()
plt.tight_layout()

if save:
    plt.savefig(figurePath + networkName.replace(" ", "")+"_SmoothOpenRates.pdf")

plt.figure()
#plt.plot(paraArray,100*specPara,label = "Specificity")
plt.plot(paraArrayMean,100*sensParaMean,'r',label = "Recall")
plt.plot(paraArrayMean,100*accParaMean,'g',label = "Accuracy")
plt.plot(paraArrayMean,100*ppvParaMean,'b',label = "Precision")
#plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
if fileName == "EvalDetectInd0Corrected.csv" :
    plt.ylim([99,100])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.ylabel("[%]")
plt.xlabel("Fault Intensity Parameter [-]")
plt.legend()
plt.title(networkName + " Open Fault Quality Measures")
plt.tight_layout()
if save:
    plt.savefig(figurePath + networkName.replace(" ", "")+"_SmoothOpenQualityMeasures.pdf")
                
#%% Accuracy and detection time vs fault occurence 
accuracyList = []
precisionList = []
recallList = [] 
detTimeList = []
if locCheck:
    for fSat in range(1,7):
        for fThr in range(1,7):
            print("Calculating for S{0}, T{1}".format(fSat,fThr))
            boolSat = Data['fSat'] == fSat 
            boolThr = Data['fThr'] == fThr 
            if networkName == "Naive Network":
                satView = 0
                boolView = Data.index % 6 == satView
                boolLoc = boolSat & boolThr & boolView
            else:
                boolLoc = boolSat & boolThr
            outLocClosed  = np.concatenate(Data[boolLoc & boolClosedFault]['outputVector'].to_numpy())
            trueLocClosed = np.concatenate(Data[boolLoc & boolClosedFault]['trueVector'].to_numpy())
            cmClosed = confusion_matrix(trueLocClosed,outLocClosed > threshold)
            accClosed = np.sum(np.diag(cmClosed))/np.sum(cmClosed)
            precClosed = cmClosed[1,1] / np.sum(cmClosed[:,1])
            recallClosed = cmClosed[1,1] / np.sum(cmClosed[1,:])
            
            outLocOpen  = np.concatenate(Data[boolLoc & boolOpenFault]['outputVector'].to_numpy())
            trueLocOpen = np.concatenate(Data[boolLoc & boolOpenFault]['trueVector'].to_numpy())
            cmOpen = confusion_matrix(trueLocOpen,outLocOpen > threshold)
            accOpen = np.sum(np.diag(cmOpen))/np.sum(cmOpen)
            precOpen = cmOpen[1,1]/np.sum(cmOpen[:,1])
            recallOpen = cmOpen[1,1]/np.sum(cmOpen[1,:])
            
            accuracyList.append((accClosed,accOpen))
            precisionList.append((precClosed,precOpen))
            recallList.append((recallClosed,recallOpen))
                              
            # tempDelayClosed = Data[boolLoc & boolClosedFault]['detTime'].to_numpy()-Data[boolLoc & boolClosedFault]['fTime'].to_numpy()
            # tempDelayOpen = Data[boolLoc & boolOpenFault]['detTime'].to_numpy()-Data[boolLoc & boolOpenFault]['fTime'].to_numpy()
            # delayClosed = np.nanmean(tempDelayClosed)
            # delayOpen = np.nanmean(tempDelayOpen)
            # detTimeList.append((delayClosed,delayOpen))
    accuracyList = np.array(accuracyList)
    precisionList = np.array(precisionList)
    recallList = np.array(recallList)
    detTimeList = np.array(detTimeList)
    #%% Bar Plot
    labels = ["Satellite {0}".format(fSat) for fSat in range(1,7)]# for fThr in range(1,7)]
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars  
    
    fig, ax = plt.subplots(figsize = [2*4.8,0.75*6.4])
    accPlot  = [100*np.mean(accuracyList[i:i+6,:])  for i in range(6)] #100*np.mean(accuracyList, axis = 1)
    precPlot = [100*np.mean(precisionList[i:i+6,:]) for i in range(6)] #100*np.mean(precisionList,axis = 1)
    recPlot  = [100*np.mean(recallList[i:i+6,:])    for i in range(6)] #100*np.mean(recallList,   axis = 1)
    ax.bar(x - 1*width, accPlot , width,label = "Accuracy",color = 'g') # Accuracy Bar plot
    ax.bar(x          , precPlot, width,label = "Precision",color = 'b') # Precision Bar Plot
    ax.bar(x + 1*width, recPlot , width,label = "Recall",color = 'r') # Recall bar plot
    plt.minorticks_on()
    ax.grid(True,which = "both")
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set(xticks = x, xticklabels = labels, ylim = [0,100],
           xlabel = "Fault Location [-]", ylabel = "Quality Measure [%]",
           title=networkName + " Detection Quality per Fault Location")
    ax.title.set_size(17)
    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)
    ax.tick_params(labelsize=14)
    plt.legend()
    fig.tight_layout()
    if save:
        plt.savefig(figurePath +networkName.replace(" ", "")+"_DetectionQuality_PerLocation.pdf")
#%% Plot Selected Reponse
run = 634
#select run
if run <0 :
    sat = -1
    thr = -1
    fty = 0
    minPara = 0
    maxPara = 0.9
    boolSat = Data['fSat'] == sat
    boolThr = Data['fThr'] == thr
    boolF = Data['fType'] == fty
    boolPara = (Data['fParam'] >= minPara) & (Data['fParam']<maxPara)
    boolSelect = boolSat & boolThr & boolF & boolPara
    run = Data[boolSelect].index[0]
    
true = Data.iloc[run]['trueVector']
fault = Data.iloc[run]['outputVector']

faultType = Data.iloc[run]['fType']
fSat = Data.iloc[run]['fSat']
fThr = Data.iloc[run]['fThr']
fParam = Data.iloc[run]['fParam']
fTime = Data.iloc[run]['fTime']
detTime = Data.iloc[run]['detTime']

fileNameRun = Data.iloc[run]['fName']
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
plt.plot(fault,'r',label = 'Network Output')
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.xlabel('time [s]')
plt.ylabel('Fault Signal')
plt.title(titleString + "")

plt.tight_layout()
plt.show()
print(fileNameRun)
if save:
    plt.savefig(figurePath + networkName.replace(" ", "") + faultString.replace(" ", "")+"_"+ str(run) +".pdf")
saveData = True
if saveData:
    aniPath = "D:\\Files\\TUDelftLocal\\Thesis\\Greenlight\\Animations\\"
    dataArray = np.vstack( (true,fault) )
    aniName = networkName.replace(" ", "") + faultString.replace(" ", "")+"_"+ str(run)
    np.savetxt(aniPath + aniName + ".csv", dataArray, delimiter=",")