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
#%% Function definitions
def detParser(fileName):
    data = []
    with open(fileName,'r') as f:
        for i,line in enumerate(f):
            if i%3 == 0:
                fName = line[:34]
                fSat = int(line[35:36])
                fThr = int(line[37:38])
                fType = int(line[39:40])
                fParam = float(line[41:])
                dat = (fName,fSat,fThr,fType,fParam)
            elif i%3 == 1:
                dataOutput = np.array(line.split(' ')).astype(float)
            else:
                dataTrue = np.array(line.split(' ')).astype(float)
                data.append((fName,fSat,fThr,fType,fParam,dataOutput,dataTrue))
    return data
def isoParser(fileName):
    data = []
    with open(fileName,'r') as f:
        counter = 0
        for line in f:
            fTest = line[:4]
            if fTest == "Test" and counter%2 == 0:
                fName = line[:34]
                fSat = int(line[35:36])
                fThr = int(line[37:38])
                fType = int(line[39:40])
                fParam = float(line[41:])
                dat = (fName,fSat,fThr,fType,fParam)
                counter += 1
            elif not fTest == "Test" and counter%2 == 1:
                dataOutput = np.array(line.split(' ')).astype(int)
                dataTrue = (6*(fSat-1)+fThr-1)*np.ones(dataOutput.shape,dtype = np.int32)
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
#%% 
# Select file to read
path=  'D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\'
network = 0
fileName = "EvalDetectInd"+str(network)+".csv"
fileNameIso = "EvalIsoInd"+str(network)+".csv"
networkName = "Detection Individual "+str(network)+" "

colNames = ['fName','fSat','fThr','fType','fParam','outputVector','trueVector']
# Read data
dataRaw = detParser(path + fileName)
isoRaw  = isoParser(path + fileNameIso)

Data = pd.DataFrame(dataRaw, columns = colNames)
isoData = pd.DataFrame(isoRaw, columns = colNames[:-2])

#Check isoData for validity
fTypeFromName = np.array( [int(fName[10]) for fName in isoData['fName'].to_numpy()])
fSatFromName = np.array( [int(fName[12]) for fName in isoData['fName'].to_numpy()])
fThrFromName = np.array( [int(fName[13]) for fName in isoData['fName'].to_numpy()])

fType = isoData['fType'].to_numpy()
fSat  = isoData['fSat'].to_numpy()
fThr  = isoData['fThr'].to_numpy()
if np.all(fTypeFromName == fType) and np.all(fSatFromName == fSat) and np.all(fThrFromName == fThrFromName):
    print("Data is consistent")


#%% Plot Confusion Matrix
boolClosedFault = Data['fType'] == 1
boolOpenFault = Data['fType'] == 2

outputCombined = np.concatenate(Data['outputVector'].to_numpy())
trueCombined = np.concatenate(Data['trueVector'].to_numpy())

outputCombinedIso = np.concatenate(isoData['outputVector'].to_numpy())
trueCombinedIso = np.concatenate(isoData['trueVector'].to_numpy())

cmIso = confusion_matrix(trueCombinedIso,outputCombinedIso)

step =0.01
detThreshold =np.arange(step,1.0,step)
cmList = np.load(path+"ConfusionMatrixDetection_thresholds.npy")
# Class Definitions
detClasses = {0:"Faultless",1:"Faulty"}
isoclasses = {0: "S1, T1",1: "S1, T2",2: "S1, T3",3: "S1, T4",4: "S1, T5",5: "S1, T6",
              6: "S2, T1",7: "S2, T2",8: "S2, T3",9: "S2, T4",10:"S2, T5",11:"S2, T6",
              12:"S3, T1",13:"S3, T2",14:"S3, T3",15:"S3, T4",16:"S3, T5",17:"S3, T6",
              18:"S4, T1",19:"S4, T2",20:"S4, T3",21:"S4, T4",22:"S4, T5",23:"S4, T6",
              24:"S5, T1",25:"S5, T2",26:"S5, T3",27:"S5, T4",28:"S5, T5",29:"S5, T6",
              30:"S6, t1",31:"S6, T2",32:"S6, T3",33:"S6, T4",34:"S6, T5",35:"S6, T6",}


#%%
tn = cmList[:,0,0]
tp = cmList[:,1,1]
fp = cmList[:,0,1]
fn = cmList[:,1,0]

spec = tn/(tn+fp)
sens = tp/(tp+fn)
acc= (tp+tn)/(tp+tn+fp+fn)


ppv = tp/(fp+tp)
npv = tn/(fn+tn)

f1 = ppv*sens/(5*ppv+sens)
fm = 2/(5/ppv+1/sens)


plt.figure()
plt.plot(detThreshold,tn,'g')
plt.plot(detThreshold,fn,'r')
plt.plot(detThreshold,tp,'y')
plt.plot(detThreshold,fp,'k')
plt.figure()
plt.plot(detThreshold,spec,label = "Specificity")
plt.plot(detThreshold,sens,label = "Sensitivity")
plt.plot(detThreshold,acc,label = "Accuracy")
plt.plot(detThreshold,ppv,label = "Positive Predictive Value")
plt.plot(detThreshold,npv,label = "Negative Predictive Value")
plt.legend()
plt.figure()
plt.plot(detThreshold,f1,label = "F1")
plt.plot(detThreshold,fm,label = "FM")
plt.legend()
#%% Determine Reaction time
threshold = detThreshold[np.argmax(fm)]
outputStack = np.stack(Data['outputVector'].to_numpy())>threshold
trueStack = np.stack(Data['trueVector'].to_numpy())

cmDet = confusion_matrix(trueStack,outputStack)
detection = np.ones((10+1,))
detection[0] = 0 
detTime = np.zeros(outputStack.shape[0])
fTime = np.zeros(outputStack.shape[0])
for i in range(outputStack.shape[0]):
    j = 0
    while j < outputStack.shape[1]-len(detection):
        det = np.all(outputStack[i,j:j+len(detection)] == detection)
        if det and not detTime[i]:
            detTime[i] = j
        fault = np.all(trueStack[i,j:j+len(detection)] == detection)
        if fault and not fTime[i]:
            fTime[i] = j
        j+=1
        if fault and det:
            break
Data['detTime'] = detTime
Data['fTime'] = fTime
#%% Compute accuracy versus fault intensity

accVsParam = np.zeros([outputStack.shape[0],5])
for i in range(outputStack.shape[0]):
    accVsParam[i,0] = Data['fParam'][i]
    conf = confusion_matrix(trueStack[i,:],outputStack[i,:])
    total =np.sum(conf)
    accVsParam[i,1] = conf[0,0]/total
    accVsParam[i,2] = conf[1,1]/total
    accVsParam[i,3] = conf[0,1]/total
    accVsParam[i,4] = conf[1,0]/total

# Gather into boxes of 0.1
gathered = np.zeros([9,4])
for i in range(9):
    para = 0.1+i/10
    
#%% Plot Comfusion Matrix

#plot_confusion_matrix(cmDet,trueCombined,outputCombined,detClasses,normalize =True,title = networkName + " Detection Confusion Matrix")
#plt.savefig(figurePath + "KalmanFaultDetection_ConfusionMatrix.pdf")
plot_confusion_matrix(cmIso,trueCombinedIso,outputCombinedIso,isoclasses,normalize =True,title = networkName + " Isolation Confusion Matrix")
#plt.savefig(figurePath + "KalmanFaultIsolation_ConfusionMatrix.pdf")

paramsClosed = Data[boolClosedFault]['fParam'].to_numpy()
paramsOpen = Data[boolOpenFault]['fParam'].to_numpy()


#%% Determine average reaction time per fault over a 0.1 fault intensity interval
delayClosed = np.zeros([9,1])
delayOpen = np.zeros([9,1])
for i in range(0,9):
    para = 0.1 + i*0.9/10
    boolPara = (Data['fParam'] >para) & (Data['fParam'] < para + 0.1)
    tempDelayClosed = Data[boolPara & boolClosedFault]['detTime'].to_numpy()-Data[boolPara & boolClosedFault]['fTime'].to_numpy()
    tempDelayOpen = Data[boolPara & boolOpenFault]['detTime'].to_numpy()-Data[boolPara & boolOpenFault]['fTime'].to_numpy()
    
    delayClosed[i] = np.mean(tempDelayClosed[tempDelayClosed>0])
    delayOpen[i] = np.mean(tempDelayOpen)
    
detDelayClosed = Data[boolClosedFault]['detTime'].to_numpy()-Data[boolClosedFault]['fTime'].to_numpy()
detDelayOpen = Data[boolOpenFault]['detTime'].to_numpy()-Data[boolOpenFault]['fTime'].to_numpy()

#%% Plotting Delay Time
boxLabels = ["[0.1,0.2)","[0.2,0.3)","[0.3,0.4)","[0.4,0.5)",
             "[0.5,0.6)","[0.6,0.7)","[0.7,0.8)","[0.8,0.9)",
             "[0.9,1.0]"]
    
plt.figure(figsize=[1.25*6.4, 1.25*4.8])
detDelayClosed_Plot = detDelayClosed
detDelayClosed_Plot[detDelayClosed < 0] = np.nan
df = pd.DataFrame(np.reshape(detDelayClosed_Plot,[9,400]).transpose(),columns = boxLabels)
sns.boxplot(data=df,showfliers = False,color="royalblue")
plt.title("Distribution of Closed Fault Detection Time")
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity Interval [-]")
bottom, top = plt.ylim()  
plt.ylim([0,top])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.show()
#plt.savefig(figurePath + "KalmanClosed_DetectionTime_BoxPlot.pdf")

plt.figure(figsize=[1.25*6.4, 1.25*4.8])
df = pd.DataFrame(np.reshape(detDelayOpen,[9,400]).transpose(),columns = boxLabels)
sns.boxplot(data=df,showfliers = False,color="royalblue")
plt.ylabel("Detection Time [s]")
plt.xlabel("Fault Intensity Interval [-]")
plt.title("Distribution of Open Fault Detection Time")
bottom, top = plt.ylim()  
plt.ylim([0,top])
plt.minorticks_on()
plt.grid(b = True, which = 'both')
plt.show()
#plt.savefig(figurePath + "KalmanOpen_DetectionTime_BoxPlot.pdf")

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
    delayClosedFine[i] = np.mean(tempDelayClosed[tempDelayClosed>0])
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
#plt.savefig(figurePath + "KalmanClosed_DetectionTime_Linear.pdf")

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
#plt.savefig(figurePath + "KalmanOpen_DetectionTime_Linear.pdf")

