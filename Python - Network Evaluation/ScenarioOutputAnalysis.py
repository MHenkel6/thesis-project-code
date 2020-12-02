# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:45:22 2020

@author: Martin
"""

#%% Import ands Dependencies
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
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
#%% Function definitions
def detParser(fileName,beginText = "Test"):
    data = []
    with open(fileName,'r') as f:
        counter = 0
        for line in f:
            fTest = line[:len(beginText)]
            if fTest == beginText and counter%3 == 0:
                fName = line[:line.find(".tfrecord")]
                fSat = int(fName[len(beginText)+3:len(beginText)+4])
                fThr = int(fName[len(beginText)+4:len(beginText)+5])
                fType = int(fName[len(beginText)+1])
                fParam = float(fName[-10:])
                counter += 1
            elif not fTest == "Test" and counter%3 == 1:
                dataOutput = np.array(line.split(' ')).astype(float)
                counter += 1
            elif not fTest == "Test" and counter%3 == 2:
                dataTrue = np.array(line.split(' ')).astype(float)
                counter += 1
                data.append((fName,fSat,fThr,fType,fParam,dataOutput,dataTrue))
    return data
def isoParser(fileName,beginText = "Test"):
    data = []
    with open(fileName,'r') as f:
        counter = 0
        for line in f:
            fTest = line[:len(beginText)]
            if fTest == beginText and counter%2 == 0:
                fName = line[:line.find(".tfrecord")]
                fSat = int(fName[len(beginText)+3:len(beginText)+4])
                fThr = int(fName[len(beginText)+4:len(beginText)+5])
                fType = int(fName[len(beginText)+1])
                fParam = float(fName[-10:])
                counter += 1
            elif not fTest == beginText and counter%2 == 1:
                dataOutput = np.array(line.split(' ')).astype(int)
                dataTrue = (6*(fSat-1)+fThr-1)*np.ones(dataOutput.shape,dtype = np.int32)
                counter += 1
                data.append((fName,fSat,fThr,fType,fParam,dataOutput,dataTrue))
    return data

def multiFailIsoParser(fileName,beginText = "Test"):
    data = []
    with open(fileName,'r') as f:
        counter = 0
        fileFirst = True
        dataTrue = []
        dataOutput = []
        fName = ""
        fSat = -1
        fThr = -1
        fType = -1
        fParam = -1
        for line in f:
            dataLine = np.nan
            dataTrueLine = np.nan
            fTest = line[:len(beginText)]
            
            if fTest == beginText:
                if not fileFirst:
                    data.append((fName,fSat,fThr,fType,fParam,np.array(dataOutput),np.array(dataTrue)))
                    dataTrue = []
                    dataOutput = []
                    counter = 0
                    
                fName = line[:line.find(".tfrecord")]
                fSat = int(fName[len(beginText)+3:len(beginText)+4])
                fThr = int(fName[len(beginText)+4:len(beginText)+5])
                fType = int(fName[len(beginText)+1])
                fParam = float(fName[-10:])
                fileFirst = False
            elif not fTest == beginText:
                dataLine = np.array(line.split(' ')).astype(float)
                dataOutput.append(dataLine)
                dataTrueLine = np.zeros(36)
                dataTrueLine[0] = 1
                if counter >= 100:
                    dataTrueLine[(6*(fSat-1)+fThr-1)] = 1
                dataTrue.append(dataTrueLine)
                counter += 1

    return data
def plot_confusion_matrix2(cm,y_true,y_pred, classes,
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
    fig.set_size_inches(8,8)
    
    if annotate:
      # Loop over data dimensions and create text annotations.
      fmt = '.0f' if normalize else 'd'
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
      fmt = '.0f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i in range(cm.shape[0]):
          for j in range(cm.shape[1]):
              ax.text(j, i, format(100*cm[i, j], fmt),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black",
                      size=textsize)
    if title:
        fig.suptitle(title,y = 0.97,fontsize = textsize + 15)
    fig.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    return ax
def plot_diff_matrix(cm,y_true,y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, 
                         annotate=True,
                         textsize = 7,
                         colorbar = True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    classes = [classes[x] for x in unique_labels(y_true, y_pred)]
    if normalize:
        #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    if normalize:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap,vmin = -0.5, vmax = 0.5)
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
      fmt = '.0f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i in range(cm.shape[0]):
          for j in range(cm.shape[1]):
              ax.text(j, i, format(100*cm[i, j], fmt),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black",
                      size=textsize)
    fig.tight_layout()
    fig.suptitle(title,y = 0.965,fontsize = 32)
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
# Select file to read
pathCM = 'D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\Data\\'
path=  'D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\Data\\Scenarios\\'
figurePath = "D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\BigFont_figures\\"

# Detection/Isolation/Both
detCheck =  False
isoCheck = True
indRun = False
scenario =  "HighNoise" # "LowIntensity" # "NavChange" # "MultiFail" # "LowThrust" #
save = False
fileNamesDet = ["EvalDetectNaive_"+scenario+".csv","EvalDetectInd0_"+scenario+".csv"]
detThreshold = [0.99,0.67]
fileNamesIso = ["EvalIsoNaive_"+scenario+".csv"]+ ["EvalIsoInd"+str(x)+"_"+scenario+".csv" for x in range(6)]+ ["EvalIsoIndCombined_"+scenario+".csv"] #"EvalIsoInd_C"+str(network)+".csv" #"EvalIsoNaive_Net2.csv" # 
networkNames = [ "Naive Network","Satellite 1"]
isoNetworkNames = [ "Naive Network","Satellite 1","Satellite 2","Satellite 3",
                "Satellite 4","Satellite 5","Satellite 6","Combined Satellite Output"]
confMatrixNames = [name.replace(" ","") + "_ConfusionMatrix.npy" for name in isoNetworkNames]
confSatMatrixNames = [name.replace(" ","") + "_SatelliteConfusionMatrix.npy" for name in isoNetworkNames]

colNames = ['fName','fSat','fThr','fType','fParam','outputVector','trueVector']
beta = 1/2

#%% Detection Evaluation
if detCheck:
    for i,fileName in enumerate(fileNamesDet):
        # Read data
        print(fileName)
        networkName = networkNames[i]
        threshold = detThreshold[i]
        
        dataRaw = detParser(path + fileName,beginText = scenario)
        Data = pd.DataFrame(dataRaw, columns = colNames)
        #Check Data for consistency
        dataFault = ~(Data['fType']==0)
        
        fTypeFromName = np.array( [int(fName[len(scenario)+1]) for fName in Data[dataFault]['fName'].to_numpy()])
        fSatFromName = np.array( [int(fName[len(scenario)+3]) for fName in Data[dataFault]['fName'].to_numpy()])
        fThrFromName = np.array( [int(fName[len(scenario)+4]) for fName in Data[dataFault]['fName'].to_numpy()])
        
        fType = Data[dataFault]['fType'].to_numpy()
        fSat  = Data[dataFault]['fSat'].to_numpy()
        fThr  = Data[dataFault]['fThr'].to_numpy()
        if scenario == "LowIntensity":
            Data[Data['fParam'] == 0]['fParam'] == 1e-5  
        if np.all(fTypeFromName == fType) and np.all(fSatFromName == fSat) and np.all(fThrFromName == fThrFromName):
            print("Data is consistent")
        
        if fileName[:15] == "EvalDetectNaive":
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
            nMean = 1
        else:
            nMean = 1
        if scenario == "MultiFail":
            #Due to a slight mistake in the file saving, the fault time
            # and therefore the fault vector is off by 100 seconds.
            for trueVec in Data['trueVector'].to_numpy():
                loc = np.where(np.array(trueVec) > 0 )[0][0]
                trueVec[loc-100:loc] = 100*[1]
        trueStack = np.stack(Data['trueVector'].to_numpy())
        outputStack = np.stack(Data['outputVector'].to_numpy())    
        boolNoFault = Data['fType'] == 0
        boolClosedFault = Data['fType'] == 1
        boolOpenFault = Data['fType'] == 2
        trueCombined = np.concatenate(Data['trueVector'].to_numpy())
        outputCombined = np.concatenate(Data['outputVector'].to_numpy())
        cmDet = confusion_matrix(trueCombined,outputCombined>threshold)
        tp = cmDet[1,1]
        tn = cmDet[0,0]
        fp = cmDet[0,1]
        fn = cmDet[1,0]
        
        acc = (tp+tn)/(tp+tn+fp+fn)
        ppv = tp/(fp+tp)
        sens = tp/(tp+fn)
        print("Accuracy:", np.round(100*acc,2))
        print("Precision:",  np.round(100*ppv,2))
        print("Recall:", np.round(100*sens,2))
        # Determine Detection Time
        detection = np.ones((5+1,))
        detection[0] = 0 
        detTime = np.empty(outputStack.shape[0])
        detTime[:] = np.nan
        fTime = np.zeros(outputStack.shape[0])
        faultAssigned = False
        detAssigned = False
        for j in range(outputStack.shape[0]):
            k = 0
            faultAssigned = False
            detAssigned = False
            while k < outputStack.shape[1]-len(detection):
                det = np.all( (outputStack[j,k:k+len(detection)]>threshold) == detection)
                if det and not detAssigned :
                    detTime[j] = k+len(detection)
                    detAssigned = True
                fault = np.all(trueStack[j,k:k+len(detection)] == detection)
                if fault and not faultAssigned:
                    fTime[j] = k
                    faultAssigned = True
                k+=1
                if faultAssigned and detAssigned:
                    break
        
        Data['detTime'] = detTime
        Data['fTime'] = fTime
        print(networkName +" Median detection time",np.nanmedian(detTime-fTime))
        detTime = Data[boolClosedFault]['detTime']
        fTime = Data[boolClosedFault]['fTime']
        detTime[detTime == 0 ] = np.nan
        print(networkName +" Median closed fault detection time",np.nanmedian(detTime-fTime))
        detTime = Data[boolOpenFault]['detTime']
        fTime = Data[boolOpenFault]['fTime']
        detTime[detTime == 0 ] = np.nan
        print(networkName +" Median open fault detection time",np.nanmedian(detTime-fTime))

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
        df = detDelayClosed_Plot.rename(columns = boxLabelsDict)
        sns.boxplot(data=df,showfliers = False,color="royalblue")
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
            plt.savefig(figurePath + scenario + "_" +networkName.replace(" ", "")+"Closed_DetectionTime_BoxPlot.pdf")
        
        fig,ax = plt.subplots(figsize=[1.25*6.4, 1.25*4.8])
        detDelayOpen_Plot = pd.DataFrame(detDelayArrayOpen).transpose()
        df = detDelayOpen_Plot.rename(columns = boxLabelsDict)
        sns.boxplot(data=df,showfliers = False,color="royalblue")
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
        plt.show()
        if save:
            plt.savefig(figurePath  + scenario + "_" + networkName.replace(" ", "")+"Open_DetectionTime_BoxPlot.pdf")
        #%% Reaction Time Line plot
        delayClosedFine = Data[boolClosedFault]['detTime'].to_numpy()-Data[boolClosedFault]['fTime'].to_numpy()
        delayOpenFine = Data[boolOpenFault]['detTime'].to_numpy()-Data[boolOpenFault]['fTime'].to_numpy()
        paraArrayClosed = Data[boolClosedFault]['fParam'].to_numpy()
        paraArrayOpen = Data[boolOpenFault]['fParam'].to_numpy()
        
        
        sortIndexClosed = np.argsort(paraArrayClosed)
        delayClosedFine = delayClosedFine[sortIndexClosed]
        paraArrayClosed = paraArrayClosed[sortIndexClosed]
        
        sortIndexOpen = np.argsort(paraArrayOpen)
        delayOpenFine = delayOpenFine[sortIndexOpen]
        paraArrayOpen = paraArrayOpen[sortIndexOpen]

        plt.figure()
        plt.plot(paraArrayClosed,delayClosedFine,label='Closed Fault Detection Time')
        plt.title("Fault Detection Time")
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
            plt.savefig(figurePath + scenario+ "_" + networkName.replace(" ", "") + "_DetectionTime_Linear.pdf")    

        #%%
        if scenario == "LowIntensity" or scenario == "LowThrust":
            accVsParam = np.zeros([outputStack[~boolNoFault].shape[0],5])
            for j in range(outputStack[~boolNoFault].shape[0]):
                accVsParam[j,0] = Data[~boolNoFault]['fParam'].iloc[j]
                conf = confusion_matrix(trueStack[j,:],outputStack[j,:]>threshold)
                accVsParam[j,1] = conf[0,0]
                accVsParam[j,2] = conf[1,1]
                accVsParam[j,3] = conf[0,1]
                accVsParam[j,4] = conf[1,0]
            accVsParamDF = pd.DataFrame(accVsParam, columns = ['fParam','tn','tp','fp','fn'])
        
            nMean = 1
            paraArray = accVsParamDF[boolClosedFault]['fParam'].to_numpy()
            tn = accVsParamDF[boolClosedFault]['tn'].to_numpy()
            tp = accVsParamDF[boolClosedFault]['tp'].to_numpy()
            fp = accVsParamDF[boolClosedFault]['fp'].to_numpy()
            fn = accVsParamDF[boolClosedFault]['fn'].to_numpy()
            
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
            plt.xscale('log' if scenario == "LowIntensity" else 'linear')
            plt.minorticks_on()
            plt.grid(b = True, which = 'both')
            plt.xlabel("Fault Intensity Parameter [-]")
            plt.ylim([0,100])
            plt.ylabel("Fraction of Dataset[%]")
            plt.title(networkName + "Closed Fault Detection Rates")
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"ClosedRates.pdf")
            
            plt.figure()
            #plt.plot(paraArray,100*specPara,label = "Specificity")
            plt.plot(paraArrayMean,100*sensParaMean,'r' ,label = "Recall")
            plt.plot(paraArrayMean,100*accParaMean,'g' ,label = "Accuracy")
            if fileName[:15] == "EvalDetectNaive":
                plt.plot(paraArrayMean,100*ppvParaMean,'.b' ,label = "Precision")
            else:
                plt.plot(paraArrayMean,100*ppvParaMean,'b' ,label = "Precision")
            #plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
            plt.xscale('log' if scenario == "LowIntensity" else 'linear')
            plt.minorticks_on()
            plt.grid(b = True, which = 'both')
            plt.ylim([-2,102])
            plt.ylabel("[%]")
            plt.xlabel("Fault Intensity Parameter [-]")
            plt.legend()
            plt.title(networkName + " Closed Fault Quality Measures")
            plt.tight_layout()
            if save:
                plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"ClosedQualityMeasures.pdf")
            
            
            paraArray = accVsParamDF[boolOpenFault]['fParam'].to_numpy()
            tn = accVsParamDF[boolOpenFault]['tn'].to_numpy()
            tp = accVsParamDF[boolOpenFault]['tp'].to_numpy()
            fp = accVsParamDF[boolOpenFault]['fp'].to_numpy()
            fn = accVsParamDF[boolOpenFault]['fn'].to_numpy()
            
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
            plt.ylim([-2,102])
            plt.xscale('log' if scenario == "LowIntensity" else 'linear')
            plt.minorticks_on()
            plt.grid(b = True, which = 'both')
            plt.ylabel("Fraction of Dataset[%]")
            plt.xlabel("Fault Intensity Parameter [-]")
            plt.ylabel("Fraction of Dataset")
            plt.title(networkName + " Open Fault Detection Rates")
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"OpenRates.pdf")
            
            plt.figure()
            #plt.plot(paraArray,100*specPara,label = "Specificity")
            plt.plot(paraArrayMean,100*sensParaMean,'r',label = "Recall")
            plt.plot(paraArrayMean,100*accParaMean,'g',label = "Accuracy")
            if fileName[:15] == "EvalDetectNaive":
                plt.plot(paraArrayMean,100*ppvParaMean,'.b' ,label = "Precision")
            else:
                plt.plot(paraArrayMean,100*ppvParaMean,'b' ,label = "Precision")
            #plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
            plt.ylim([-2,102])
            plt.xscale('log' if scenario == "LowIntensity" else 'linear')
            plt.minorticks_on()
            plt.grid(b = True, which = 'both')
            plt.ylabel("[%]")
            plt.xlabel("Fault Intensity Parameter [-]")
            plt.legend()
            plt.title(networkName + " Open Fault Quality Measures")
            plt.tight_layout()
            if save:
                plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"OpenQualityMeasures.pdf")
        #%%
        # if scenario == "LowThrust":
        #     accVsParam = np.zeros([outputStack[~boolNoFault].shape[0],5])
        #     for j in range(outputStack[~boolNoFault].shape[0]):
        #         accVsParam[j,0] = Data[~boolNoFault]['fParam'].iloc[j]
        #         conf = confusion_matrix(trueStack[j,:],outputStack[j,:]>threshold)
        #         accVsParam[j,1] = conf[0,0]
        #         accVsParam[j,2] = conf[1,1]
        #         accVsParam[j,3] = conf[0,1]
        #         accVsParam[j,4] = conf[1,0]
        #     accVsParamDF = pd.DataFrame(accVsParam, columns = ['fParam','tn','tp','fp','fn'])
        
        
        #     paraArray = accVsParamDF[boolClosedFault]['fParam'].to_numpy()
        #     tn = accVsParamDF[boolClosedFault]['tn'].to_numpy()
        #     tp = accVsParamDF[boolClosedFault]['tp'].to_numpy()
        #     fp = accVsParamDF[boolClosedFault]['fp'].to_numpy()
        #     fn = accVsParamDF[boolClosedFault]['fn'].to_numpy()
            
        #     total = tn+tp+fp+fn
        #     tnPara = tn / total
        #     tpPara = tp / total
        #     fpPara = fp / total
        #     fnPara = fn / total 
        #     # Sort arrays
        #     sortIndex = np.argsort(paraArray)
        #     tnPara = tnPara[sortIndex]
        #     tpPara = tpPara[sortIndex]
        #     fpPara = fpPara[sortIndex]
        #     fnPara = fnPara[sortIndex]
        #     paraArray = paraArray[sortIndex]
        #     specPara = tnPara/(tnPara+fpPara)
        #     sensPara = tpPara/(tpPara+fnPara) #sensitivity aka recall
        #     accPara = (tpPara+tnPara)/(tpPara+tnPara+fpPara+fnPara)
        #     totalPara = tpPara + tnPara + fpPara + fnPara
            
        #     ppvPara = tpPara/(fpPara + tpPara) # positive predictive value, aka precision
        #     npvPara = tnPara/(fnPara + tnPara)
        #     f1Para = (1+beta**2)*ppvPara*sensPara/((beta**2*ppvPara)+sensPara)
            
        #      # Calculate mean over 5 elements
        #     tnParaMean = np.array([np.nanmean(tnPara[nMean*x:nMean*x+nMean]) for x in range(len(tnPara)//nMean)])
        #     fnParaMean = np.array([np.nanmean(fnPara[nMean*x:nMean*x+nMean]) for x in range(len(fnPara)//nMean)])
        #     tpParaMean = np.array([np.nanmean(tpPara[nMean*x:nMean*x+nMean]) for x in range(len(tpPara)//nMean)])
        #     fpParaMean = np.array([np.nanmean(fpPara[nMean*x:nMean*x+nMean]) for x in range(len(fpPara)//nMean)])
            
        #     sensParaMean  = np.array([np.nanmean(sensPara[nMean*x:nMean*x+nMean]) for x in range(len(sensPara)//nMean)])
        #     ppvParaMean   = np.array([np.nanmean(ppvPara[nMean*x:nMean*x+nMean]) for x in range(len(ppvPara)//nMean)])
        #     accParaMean   = np.array([np.nanmean(accPara[nMean*x:nMean*x+nMean]) for x in range(len(accPara)//nMean)])
        #     paraArrayMean = np.array([np.nanmean(paraArray[nMean*x:nMean*x+nMean]) for x in range(len(paraArray)//nMean)])
        #     # Moving window Average
        #     tnParaMean = running_mean(tnPara, nMean)
        #     fnParaMean = running_mean(fnPara, nMean)
        #     tpParaMean = running_mean(tpPara, nMean)
        #     fpParaMean = running_mean(fpPara, nMean)
            
        #     sensParaMean  = running_mean(sensPara, nMean)
        #     ppvParaMean   = running_mean(ppvPara, nMean)
        #     accParaMean   = running_mean(accPara, nMean)
        #     paraArrayMean = running_mean(paraArray, nMean)
            
        #     plt.figure()
        #     plt.plot(paraArrayMean,100*tnParaMean,'g',label = 'True Negative')
        #     plt.plot(paraArrayMean,100*fnParaMean,'r',label = 'False Negative')
        #     plt.plot(paraArrayMean,100*tpParaMean,'y',label = 'True Positive')
        #     plt.plot(paraArrayMean,100*fpParaMean,'k',label = 'False Positive')
    
        #     plt.minorticks_on()
        #     plt.grid(b = True, which = 'both')
        #     plt.xlabel("Fault Intensity Parameter [-]")
        #     plt.ylim([0,100])
        #     plt.ylabel("Fraction of Dataset[%]")
        #     plt.title(networkName + " Closed Fault Detection Rates")
        #     plt.legend()
        #     if save:
        #         plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"ClosedRates.pdf")
            
        #     plt.figure()
        #     #plt.plot(paraArray,100*specPara,label = "Specificity")
        #     plt.plot(paraArrayMean,100*sensParaMean,'r' ,label = "Recall")
        #     plt.plot(paraArrayMean,100*accParaMean,'g' ,label = "Accuracy")
        #     plt.plot(paraArrayMean,100*ppvParaMean,'b' ,label = "Precision")
        #     #plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
        #     plt.minorticks_on()
        #     plt.grid(b = True, which = 'both')
        #     plt.ylim([-2,102])
        #     plt.ylabel("[%]")
        #     plt.xlabel("Fault Intensity Parameter [-]")
        #     plt.legend()
        #     plt.title(networkName + " Closed Fault Quality Measures")
        #     if save:
        #         plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"ClosedQualityMeasures.pdf")
            
            
        #     paraArray = accVsParamDF[boolOpenFault]['fParam'].to_numpy()
        #     tn = accVsParamDF[boolOpenFault]['tn'].to_numpy()
        #     tp = accVsParamDF[boolOpenFault]['tp'].to_numpy()
        #     fp = accVsParamDF[boolOpenFault]['fp'].to_numpy()
        #     fn = accVsParamDF[boolOpenFault]['fn'].to_numpy()
            
        #     total = tn+tp+fp+fn
        #     tnPara = tn / total
        #     tpPara = tp / total
        #     fpPara = fp / total
        #     fnPara = fn / total 
        #     # Sort arrays
        #     sortIndex = np.argsort(paraArray)
        #     tnPara = tnPara[sortIndex]
        #     tpPara = tpPara[sortIndex]
        #     fpPara = fpPara[sortIndex]
        #     fnPara = fnPara[sortIndex]
        #     paraArray = paraArray[sortIndex]
        #     specPara = tnPara/(tnPara+fpPara)
        #     sensPara = tpPara/(tpPara+fnPara) #sensitivity aka recall
        #     accPara = (tpPara+tnPara)/(tpPara+tnPara+fpPara+fnPara)
        #     totalPara = tpPara + tnPara + fpPara + fnPara
            
        #     ppvPara = tpPara/(fpPara + tpPara) # positive predictive value, aka precision
        #     npvPara = tnPara/(fnPara + tnPara)
        #     f1Para = (1+beta**2)*ppvPara*sensPara/((beta**2*ppvPara)+sensPara)
            
        #      # Calculate mean over 5 elements
        #     # tnParaMean = np.array([np.nanmean(tnPara[nMean*x:nMean*x+nMean]) for x in range(len(tnPara)//nMean)])
        #     # fnParaMean = np.array([np.nanmean(fnPara[nMean*x:nMean*x+nMean]) for x in range(len(fnPara)//nMean)])
        #     # tpParaMean = np.array([np.nanmean(tpPara[nMean*x:nMean*x+nMean]) for x in range(len(tpPara)//nMean)])
        #     # fpParaMean = np.array([np.nanmean(fpPara[nMean*x:nMean*x+nMean]) for x in range(len(fpPara)//nMean)])
            
        #     # sensParaMean  = np.array([np.nanmean(sensPara[nMean*x:nMean*x+nMean]) for x in range(len(sensPara)//nMean)])
        #     # ppvParaMean   = np.array([np.nanmean(ppvPara[nMean*x:nMean*x+nMean]) for x in range(len(ppvPara)//nMean)])
        #     # accParaMean   = np.array([np.nanmean(accPara[nMean*x:nMean*x+nMean]) for x in range(len(accPara)//nMean)])
        #     # paraArrayMean = np.array([np.nanmean(paraArray[nMean*x:nMean*x+nMean]) for x in range(len(paraArray)//nMean)])
        #     # Moving window Average
        #     tnParaMean = running_mean(tnPara, nMean)
        #     fnParaMean = running_mean(fnPara, nMean)
        #     tpParaMean = running_mean(tpPara, nMean)
        #     fpParaMean = running_mean(fpPara, nMean)
            
        #     sensParaMean  = running_mean(sensPara, nMean)
        #     ppvParaMean   = running_mean(ppvPara, nMean)
        #     accParaMean   = running_mean(accPara, nMean)
        #     paraArrayMean = running_mean(paraArray, nMean)
        #     plt.figure()
        #     plt.plot(paraArrayMean,100*tnParaMean,'g',label = 'True Negative')
        #     plt.plot(paraArrayMean,100*fnParaMean,'r',label = 'False Negative')
        #     plt.plot(paraArrayMean,100*tpParaMean,'y',label = 'True Positive')
        #     plt.plot(paraArrayMean,100*fpParaMean,'k',label = 'False Positive')
        #     plt.ylim([0,100])
        #     plt.minorticks_on()
        #     plt.grid(b = True, which = 'both')
        #     plt.ylabel("Fraction of Dataset[%]")
        #     plt.xlabel("Fault Intensity Parameter [-]")
        #     plt.ylabel("Fraction of Dataset")
        #     plt.title(networkName + " Open Fault Detection Rates")
        #     plt.legend()
        #     if save:
        #         plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"OpenRates.pdf")
            
        #     plt.figure()
        #     #plt.plot(paraArray,100*specPara,label = "Specificity")
        #     plt.plot(paraArrayMean,100*sensParaMean,'r',label = "Recall")
        #     plt.plot(paraArrayMean,100*accParaMean,'g',label = "Accuracy")
        #     plt.plot(paraArrayMean,100*ppvParaMean,'b',label = "Precision")
        #     #plt.plot(paraArray,100*npvPara,label = "Negative Predictive Value")
        #     plt.ylim([-2,102])
        #     plt.minorticks_on()
        #     plt.grid(b = True, which = 'both')
        #     plt.ylabel("[%]")
        #     plt.xlabel("Fault Intensity Parameter [-]")
        #     plt.legend()
        #     plt.title(networkName + " Open Fault Quality Measures")
        #     if save:
        #         plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"OpenQualityMeasures.pdf")
                                
        #select run
        if indRun:
            if scenario == "MultiFail":
                sat = 2
                thr = 3
                fty = 2
            else:
                sat = 1
                thr = 1
                fty = 1
            minPara = 0
            maxPara = 0.9
            boolSat = Data['fSat'] == sat
            boolThr = Data['fThr'] == thr
            boolF = Data['fType'] == fty
            boolPara = (Data['fParam'] >= minPara) & (Data['fParam']<maxPara)
            boolSelect = boolSat & boolThr & boolF & boolPara
            
            run = Data[boolSelect].index[0]+1
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
            plt.title(networkName + " " + titleString)
            plt.tight_layout()
            plt.show()
            print(fileNameRun)
            if save:
                plt.savefig(figurePath +scenario + "_" + networkName.replace(" ", "")+faultString.replace(" ", "")+"_"+ str(run) +".pdf")
# %% Isolation Analysis
cmList = [] 
accList = []
closedAccList = []
openAccList = []
recList = [] 
precList = [] 
start = 4

if isoCheck:
    for i,fileNameIso in enumerate(fileNamesIso[start:start+1]):
        print(fileNameIso)

        networkName = isoNetworkNames[i+start]
        if scenario == "MultiFail":
            isoRaw = multiFailIsoParser(path + fileNameIso,beginText = scenario)
        else:
            isoRaw = isoParser(path + fileNameIso,beginText=scenario)
        
        isoData = pd.DataFrame(isoRaw, columns = colNames)
            
        #Check isoData for validity
        fTypeFromName = np.array( [int(fName[len(scenario)+1]) for fName in isoData['fName'].to_numpy()])
        fSatFromName = np.array( [int(fName[len(scenario)+3]) for fName in isoData['fName'].to_numpy()])
        fThrFromName = np.array( [int(fName[len(scenario)+4]) for fName in isoData['fName'].to_numpy()])
        
        fType = isoData['fType'].to_numpy()
        fSat  = isoData['fSat'].to_numpy()
        fThr  = isoData['fThr'].to_numpy()
        if np.all(fTypeFromName == fType) and np.all(fSatFromName == fSat) and np.all(fThrFromName == fThrFromName):
            print("Data is consistent")
        
        boolClosedFault = isoData['fType'] == 1
        boolOpenFault = isoData['fType'] == 2
        if scenario == "MultiFail":
            outputList = []
            for j in range(len(isoData['outputVector'])):
                output = isoData['outputVector'].iloc[j]
                isoMax = np.argmax(output,axis = 1)
                outputList.append(isoMax)
            outputCombinedIso = np.concatenate(outputList)   
            trueList = []
            for trueState in isoData['trueVector']:
                # Use as true output where the matrix is not zero, except or the first
                # column as it is always nonzero. Since we are ignoring the first column
                # we need to add 1 to the output again to account for the offset
                trueVec = np.zeros(trueState.shape[0])
                trueVec[np.where(trueState[:,1:])[0]] = np.where(trueState[:,1:])[1]+1
                trueList.append(trueVec)
            trueCombinedIso = np.concatenate(trueList)
        else:
            outputCombinedIso = np.concatenate(isoData['outputVector'].to_numpy())
            trueCombinedIso = np.concatenate(isoData['trueVector'].to_numpy())
        cmIso = confusion_matrix(trueCombinedIso,outputCombinedIso,labels = range(36))
        cmList.append(cmIso)
        if save:
            np.save(path + scenario + networkName.replace(" ", "")+"_ConfusionMatrix",cmIso)
        # Class Definition
        isoclasses = {0: "S1, T1",1: "S1, T2",2: "S1, T3",3: "S1, T4",4: "S1, T5",5: "S1, T6",
                      6: "S2, T1",7: "S2, T2",8: "S2, T3",9: "S2, T4",10:"S2, T5",11:"S2, T6",
                      12:"S3, T1",13:"S3, T2",14:"S3, T3",15:"S3, T4",16:"S3, T5",17:"S3, T6",
                      18:"S4, T1",19:"S4, T2",20:"S4, T3",21:"S4, T4",22:"S4, T5",23:"S4, T6",
                      24:"S5, T1",25:"S5, T2",26:"S5, T3",27:"S5, T4",28:"S5, T5",29:"S5, T6",
                      30:"S6, t1",31:"S6, T2",32:"S6, T3",33:"S6, T4",34:"S6, T5",35:"S6, T6",}
        
        if scenario == "LowThrust" or scenario == "LowIntensity":
            regCM = np.load(pathCM + confMatrixNames[i])
            regCMnorm = np.zeros(regCM.shape)
            cmIsoNorm = np.zeros(cmIso.shape)
            for j,row in enumerate(regCM):
                regCMnorm[j,:] = row/sum(row)
            for j,row in enumerate(cmIso):
                cmIsoNorm[j,:] = row/sum(row)
            cmDiff = cmIsoNorm[0,:]-regCMnorm[0,:]
        #     color = []
        #     plt.figure()
        #     plt.bar(range(36),cmDiff,color = np.where(cmDiff > 0,['b' for _ in cmDiff],['r' for _ in cmDiff]))
        #     plt.minorticks_on()
        #     plt.grid(b = True, which = 'both')      
        #     plt.xlabel("Satellite Classification")
        #     plt.ylabel("Difference in Classification [%]")
       
            isoAccVsParam = np.zeros([len(isoData),2])
            for j in range(len(isoData)):
                isoAccVsParam[j,0] = isoData['fParam'].iloc[j]
                output = isoData['outputVector'].iloc[j]
                true = isoData['trueVector'].iloc[j]
                isoCorrect = output == true
                isoAccVsParam[j,1] = sum(isoCorrect)/len(isoCorrect)

            isoAccVsParam = pd.DataFrame(isoAccVsParam, columns = ['fParam','isoAcc'])
            
            paraArray = isoAccVsParam['fParam'].to_numpy()
            isoAccArray = isoAccVsParam['isoAcc'].to_numpy()

            sortIndex = np.argsort(paraArray)
            isoAccArray = isoAccArray[sortIndex]
            paraArray = paraArray[sortIndex]
            
            nMean = 1
            paraArrayMean = running_mean(paraArray,nMean)
            isoAccArrayMean = running_mean(isoAccArray,nMean)

            # Closed Faults
            paraArrayClosed = isoAccVsParam[boolClosedFault]['fParam'].to_numpy()
            isoAccArrayClosed = isoAccVsParam[boolClosedFault]['isoAcc'].to_numpy()

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
            # plt.plot(paraArrayMean,100*isoAccArrayMean,label = "Average Isolation Accuracy")
            plt.plot(paraArrayMeanClosed,100*isoAccArrayMeanClosed,label = "Closed Fault Isolation Accuracy")
            plt.plot(paraArrayMeanOpen  ,100*isoAccArrayMeanOpen  ,label = "Open Fault Isolation Accuracy")
            plt.ylim([0,100])
            plt.xscale('log' if scenario == "LowIntensity" else 'linear')
            # plt.xlim(1e-2,1e-1)
            plt.minorticks_on()
            plt.grid(b = True, which = 'both')
            plt.ylabel("Isolation Accuracy [%]")
            plt.xlabel("Fault Intensity Parameter [-]")
            plt.title(networkName + " Isolation Accuracy")
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"IsoAccBoth_FaultIntensity.pdf")

      
        else:
            regCM = np.load(pathCM + confMatrixNames[i+start])
            regCMnorm = np.zeros(regCM.shape)
            cmIsoNorm = np.zeros(cmIso.shape)
            for j,row in enumerate(regCM):
                regCMnorm[j,:] = row/sum(row)
            for j,row in enumerate(cmIso):
                cmIsoNorm[j,:] = row/sum(row)
            cmDiff = cmIsoNorm-regCMnorm
            plot_diff_matrix(cmDiff,trueCombinedIso,outputCombinedIso,isoclasses,
                             textsize = 9, normalize =True,
                             title = networkName + " Isolation Confusion Matrix",
                             cmap = plt.cm.seismic_r)
            if save:
                plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"_DifferenceMatrix.pdf")
            if scenario == "NavChange" or scenario == "NavChangeV2":
                isoSatClasses = {0:"Sat 1", 1:"Sat 2", 2:"Sat 3", 3:"Sat 4", 4:"Sat 5", 5:"Sat 6"}

                cmIsoSat = confusion_matrix(trueCombinedIso//6,outputCombinedIso//6)

                regCMSat = np.load(pathCM + confSatMatrixNames[i+start])
                regCMSatNorm = np.zeros(regCMSat.shape)
                cmIsoSatNorm = np.zeros(cmIsoSat.shape)
                for j,row in enumerate(regCMSat):
                    regCMSatNorm[j,:] = row/sum(row)
                for j,row in enumerate(cmIsoSat):
                    cmIsoSatNorm[j,:] = row/sum(row)
                cmDiffSat = cmIsoSatNorm-regCMSatNorm
                plot_diff_matrix(cmDiffSat,trueCombinedIso//6,outputCombinedIso//6,isoSatClasses,
                                 textsize = 30,normalize ="row",title = networkName + " Satellite Isolation Confusion Matrix",
                                 cmap = plt.cm.seismic_r)
                if save:
                    plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"_SatelliteDifferenceMatrix.pdf")

                
            
        # plot_confusion_matrix(cmIso,trueCombinedIso,outputCombinedIso,isoclasses,normalize =True,title = networkName + " Isolation Confusion Matrix")
        # if save:
        #     plt.savefig(figurePath + scenario + "_" + networkName.replace(" ", "")+"_FaultIsolation_ConfusionMatrix.pdf")
        if scenario == "MultiFail":
            isoCorrect = np.logical_or(outputCombinedIso == trueCombinedIso,
                                       outputCombinedIso == 0)
            cmIsoadapt = np.copy(cmIso)
            cmIsoadapt[np.diag_indices(cmIsoadapt.shape[0])] += cmIsoadapt[:,0]
            cmIsoadapt[0,0] /= 2
            cmIsoadapt[1:,0] = 0 
            cmRow = cmIsoadapt.astype('float') / cmIsoadapt.sum(axis=1)[:, np.newaxis]
            isoRec = np.nanmean(np.diag(cmRow))

            # Isolation Precision (column normalized confusion matrix)
            cmCol = cmIsoadapt.astype('float') / cmIsoadapt.sum(axis=0)[np.newaxis,:]
            isoPrec = np.nanmean(np.diag(cmCol))
            
        else:
            isoCorrect = outputCombinedIso == trueCombinedIso
             # Isolation Recall (row normalized confusion matrix)
            cmRow = cmIso.astype('float') / cmIso.sum(axis=1)[:, np.newaxis]
            isoRec = np.nanmean(np.diag(cmRow))

            # Isolation Precision (column normalized confusion matrix)
            cmCol = cmIso.astype('float') / cmIso.sum(axis=0)[np.newaxis,:]
            isoPrec = np.nanmean(np.diag(cmCol))
        
        acc = sum(isoCorrect)/len(isoCorrect)
        accList.append(acc)
        print(networkName + " Isolation Accuracy: ", np.round(100*acc,2))
        print(networkName + " Isolation Precision: ", np.round(100*isoPrec,2))
        precList.append(isoPrec)
        print(networkName + " Isolation Recall: ", np.round(100*isoRec,2))  
        recList.append(isoRec)

        
       
        # Isolation Accuracy for closed faults 
        if scenario == "MultiFail":
            outputList = []
            trueList = []
            for j in range(len(isoData['outputVector'])):
                output = isoData['outputVector'].iloc[j]
                isoMax = np.argmax(output,axis = 1)
                outputList.append(isoMax)
            outputCombinedIso = np.concatenate(outputList)   
            
            for trueState in isoData['trueVector']:
                # Use as true output where the matrix is nonzero, except or the first
                # column as it is always nonzero. Since we are ignoring the first column
                # we need to add 1 to the output again to account for the offset
                trueVec = np.zeros(trueState.shape[0])
                trueVec[np.where(trueState[:,1:])[0]] = np.where(trueState[:,1:])[1]+1
                trueList.append(trueVec)
            trueCombinedIso = np.concatenate(trueList)
            isoCorrect = np.logical_or(outputCombinedIso == trueCombinedIso,
                                       outputCombinedIso == 0)
        else:
            outputCombinedIso = np.concatenate(isoData[boolClosedFault]['outputVector'].to_numpy())
            trueCombinedIso = np.concatenate(isoData[boolClosedFault]['trueVector'].to_numpy())
            isoCorrect = outputCombinedIso == trueCombinedIso
            
        closedAcc = sum(isoCorrect)/len(isoCorrect)   
        closedAccList.append(closedAcc)
        print(networkName + " Isolation Accuracy Closed Fault:", np.round(100*closedAcc,2))
        # Isolation Accuracy for open faults 
        if scenario == "MultiFail":
            outputList = []
            trueList = []
            for j in range(len(isoData[boolOpenFault]['outputVector'])):
                output = isoData[boolOpenFault]['outputVector'].iloc[j]
                isoMax = np.argmax(output,axis = 1)
                outputList.append(isoMax)
            outputCombinedIso = np.concatenate(outputList)   
            
            for trueState in isoData[boolOpenFault]['trueVector']:
                # Use as true output where the matrix is not zero, except or the first
                # column as it is always nonzero. Since we are ignoring the first column
                # we need to add 1 to the output again to account for the offset
                trueVec = np.zeros(trueState.shape[0])
                trueVec[np.where(trueState[:,1:])[0]] = np.where(trueState[:,1:])[1]+1
                trueList.append(trueVec)
            trueCombinedIso = np.concatenate(trueList)
            isoCorrect = np.logical_or(outputCombinedIso == trueCombinedIso,
                                       outputCombinedIso == 0)
        else:
            outputCombinedIso = np.concatenate(isoData[boolOpenFault]['outputVector'].to_numpy())
            trueCombinedIso = np.concatenate(isoData[boolOpenFault]['trueVector'].to_numpy())
            isoCorrect = outputCombinedIso == trueCombinedIso
        openAcc = sum(isoCorrect)/len(isoCorrect)   
        openAccList.append(closedAcc)
        print(networkName + " Isolation Accuracy Open Fault:", np.round(100*openAcc,2))
    
        isoSatClasses = {0:"Sat 1", 1:"Sat 2", 2:"Sat 3", 3:"Sat 4", 4:"Sat 5", 5:"Sat 6"}
        cmIsoSat = confusion_matrix(trueCombinedIso//6,outputCombinedIso//6)
        
        #%% Create Multiple fault plotss 
        if scenario == "MultiFail":
            plot_confusion_matrix(cmIso,trueCombinedIso,outputCombinedIso,isoclasses,normalize ="row",title = networkName + " Isolation Confusion Matrix",textsize=12)
            if save:
                plt.savefig(figurePath + scenario + networkName.replace(" ", "")+"_ConfusionMatrix.pdf")
            #%%
            accSecondary = []
            ratioListNormal = []
            ratioList = []
            for j in range(isoData.shape[0]):
                netOutput = isoData['outputVector'].iloc[j]
                topChoicesList = []

                for i,netRow in enumerate(netOutput):
                    topChoices = np.argsort(-netRow)[:2] # Select top two indices
                    ratio = netRow[np.argsort(-netRow)[0]]/netRow[np.argsort(-netRow)[1]]
                    topChoicesList.append(topChoices)
                    if i<100:
                        ratioListNormal.append(ratio)
                    else:
                        ratioList.append(ratio)
                
                topChoicesList = np.array(topChoicesList)
                trueState = isoData['trueVector'].iloc[j]
                trueVec = np.zeros(trueState.shape[0])
                trueVec[np.where(trueState[:,1:])[0]] = np.where(trueState[:,1:])[1]+1
                accSecondary.append((np.sum(topChoicesList[100:,:] == trueVec[101]),len(topChoicesList[100:,0])))
            
            ratioList = np.array(ratioList)
            ratioListNormal = np.array(ratioListNormal)
            ratioPlot = pd.DataFrame([ratioListNormal,ratioList]).transpose()
            df = ratioPlot.rename(columns = {0:"Ratio in single-fault case",1:"Ratio in multi-fault scenario"})
            
            #%%
            plt.figure(figsize=[1.4*6.4, 1.0*4.8])
            sns.boxplot(data=df,showfliers = False,color="royalblue")
            plt.title(networkName + " ratio of largest component to second largest")
            plt.ylabel("Ratio between fault probabilities [-]")
            plt.minorticks_on()
            plt.grid(b = True, which = 'both')
            plt.tight_layout()
            if save:
                plt.savefig(figurePath +scenario + networkName.replace(" ", "")+"RatioBoxPlot.pdf")
            accSecondary = np.array(accSecondary)
            #%%
            print("Top two accuracy: ",np.round(100*np.sum(accSecondary[:,0])/np.sum(accSecondary[:,1]),2))
            # Select highest 3 components and plot their 
            run = 150
            netOutput = isoData['outputVector'].iloc[run]
            topChoicesList = []
            ratioList = []
            for netRow in netOutput:
                topChoices = np.argsort(-netRow)[:3] # Select top three indices
                ratio = netRow[np.argsort(-netRow)[0]]/netRow[np.argsort(-netRow)[1]]
                topChoicesList.append(topChoices)
                ratioList.append(ratio)
            topChoicesList = np.array(topChoicesList)
            ratioList= np.array(ratioList)
            trueState = isoData['trueVector'].iloc[run]
            trueVec = np.zeros(trueState.shape[0])
            trueVec[np.where(trueState[:,1:])[0]] = np.where(trueState[:,1:])[1]+1
            
            plt.figure()
            plt.plot(np.zeros_like(trueState),'*b')
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
        # select run
        if indRun:
            if scenario == "LowIntensity" or scenario == "LowThrust":
                sat = 1
                thr = 1
            else:
                sat = 2
                thr = 3
                
            fty = 2
            minPara = 0
            maxPara = 0.9
            boolSat = isoData['fSat'] == sat
            boolThr = isoData['fThr'] == thr
            boolF = isoData['fType'] == fty
            boolPara = (isoData['fParam'] >= minPara) & (isoData['fParam']<maxPara)
            boolSelect = boolSat & boolThr & boolF & boolPara
            
            run = isoData[boolSelect].index[0]
            true = isoData.iloc[run]['trueVector']
            fault = isoData.iloc[run]['outputVector']
            
            faultType = isoData.iloc[run]['fType']
            fSat = isoData.iloc[run]['fSat']
            fThr = isoData.iloc[run]['fThr']
            fParam = isoData.iloc[run]['fParam']
        
            
            fileNameRun = isoData.iloc[run]['fName']
            if faultType == 1:
                faultString = "Closed Fault" if faultType == 1 else "Open Fault"
                titleString = faultString + " in Satellite " + str(fSat) + " thruster "+str(fThr) + ", intensity {:.2f}".format(fParam) 
            elif faultType == 2:
                faultString = "Closed Fault" if faultType == 1 else "Open Fault"
                titleString = faultString + " in Satellite " + str(fSat) + " thruster "+str(fThr) + ", intensity {:.2f}".format(fParam) 
            else:
                faultString = "Faultless"
                titleString = "Faultless Case"
            
            plt.figure()
            plt.plot(true,'.b',label = 'True Fault Signal')
            plt.plot(fault,'.r',label = 'Network Output')
            plt.ylim([-1,36])
            plt.minorticks_on()
            plt.grid(b = True, which = 'both')
            plt.xlabel('time [s]')
            plt.ylabel('Fault Signal')
            plt.title(networkName + " " + titleString)
            plt.show()
            plt.tight_layout()
            print(fileNameRun)
            if save:
                plt.savefig(figurePath +scenario + "_TimeSeries_" + networkName.replace(" ", "")+faultString.replace(" ", "")+"_"+ str(run) +".pdf")
    #%%
    print("Accuracies")
    print(np.round(100*accList[0],decimals = 2))
    print(np.round(100*np.mean(accList[1-start:-1]),2),np.round(100*np.std(accList[1-start:-1]),2))
    print(np.round(100*accList[-1],2))
    print("Precisions")
    print(np.round(100*precList[0],2))
    print(np.round(100*np.mean(precList[1-start:-1]),2),np.round(100*np.std(precList[1-start:-1]),2))
    print(np.round(100*precList[-1],decimals = 2))
    print("Recalls")
    print(np.round(100*recList[0],2))
    print(np.round(100*np.mean(recList[1-start:-1]),2),np.round(100*np.std(recList[1-start:-1]),2))
    print(np.round(100*recList[-1],2))