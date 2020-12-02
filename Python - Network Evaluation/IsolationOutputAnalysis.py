# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:45:22 2020

@author: Martin
"""

#%% Import ands Dependencies
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
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
def split_equal_value(iterable):
    lastValue = iterable[0]
    splitList = []
    lenList = []
    lenCounter = 1
    iterLast = 0
    for i,value in enumerate(iterable[1:]):
        if lastValue == value:
            lenCounter += 1
        else:
            splitList.append(iterable[iterLast:i+1])
            iterLast = i+1
            lastValue = value
            lenList.append(lenCounter)
            lenCounter = 1
    splitList.append(iterable[iterLast:])
    lenList.append(lenCounter)
    return splitList, lenList
#%% 
# Select file to read
path=  'D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\Data\\'
figurePath = "D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\BigFont_figures\\"

#for network in range(6):
save = False
saveCM = False
load = True
indRun = True
dataset0test = False
analyzeCM = True

networkSelect = "Combined"
if networkSelect == "Naive":
    networkName = "Naive Network"
    fileNameIso = ["EvalIsoNaive.csv" ]
elif networkSelect == "Naive2":
    networkName = "Naive Network 2"
    fileNameIso =["EvalIsoNaive_Net2.csv"]
elif networkSelect == "Combined":
    networkName = "Combined Satellite Output"
    fileNameIso = ["EvalIsoIndCombined_C.csv"] # 
elif networkSelect == "Individual":
    if not dataset0test:
        fileNameIso = ["EvalIsoInd_C"+str(network)+".csv" for network in range(6)] 
        networkNames = ["Satellite "+str(i+1) for i in range(6)] 
    else:
        fileNameIso = ["EvalIsoInd_C"+str(network)+"_Dataset0.csv" for network in range(5,6)] 
        networkNames = ["Satellite "+str(i+1) + " Dataset 1" for i in range(5,6)] 

elif networkSelect == "All":
    fileNameIso = ["EvalIsoNaive.csv"] + ["EvalIsoInd_C"+str(network)+".csv" for network in range(6)]  + ["EvalIsoIndCombined_C.csv"] # 
    networkNames = ["Naive Network"] + ["Satellite "+str(i+1) for i in range(6)] + ["Combined Satellite Output"]
elif networkSelect == "NaiveTest":
    epochArray = np.array(list(range(3)) + list(range(10,20)))+1
    fileNameIso = ["NaiveTest_Epoch"+str(epoch)+".csv" for epoch in epochArray]
    networkNames = ["Naive Network at Epoch "+str(epoch)for epoch in epochArray]
    path=  'D:\\Files\\TUDelftLocal\\Thesis\\Software\\Evaluation\\Data\\NaiveTest\\'
#%% Analysis
accList = []
recList = []
precList = []
closedMList = []
openMList = []
isoMaxList = []
for i,fileName in enumerate(fileNameIso):
    if networkSelect == "Individual" or networkSelect == "All"  or networkSelect == "NaiveTest":
        networkName =  networkNames[i]
        

    colNames = ['fName','fSat','fThr','fType','fParam','outputVector','trueVector']
    # Read data
    isoRaw  = isoParser(path + fileName)
    
    isoData = pd.DataFrame(isoRaw, columns = colNames)
    
    #Check isoData for validity
    fTypeFromName = np.array( [int(fName[10]) for fName in isoData['fName'].to_numpy()])
    fSatFromName = np.array( [int(fName[12]) for fName in isoData['fName'].to_numpy()])
    fThrFromName = np.array( [int(fName[13]) for fName in isoData['fName'].to_numpy()])
    
    fType = isoData['fType'].to_numpy()
    fSat  = isoData['fSat'].to_numpy()
    fThr  = isoData['fThr'].to_numpy()
    if np.all(fTypeFromName == fType) and np.all(fSatFromName == fSat) and np.all(fThrFromName == fThrFromName):
        print("Data is consistent")
    
    

    boolClosedFault = isoData['fType'] == 1
    boolOpenFault = isoData['fType'] == 2
    
    if analyzeCM:
        outputCombinedIso = np.concatenate(isoData['outputVector'].to_numpy())
        trueCombinedIso = np.concatenate(isoData['trueVector'].to_numpy())
        if load:
            cmIso = np.load(path + networkName.replace(" ", "")+"_ConfusionMatrix.npy")
        else:
            cmIso = confusion_matrix(trueCombinedIso,outputCombinedIso)
    
        if save and not load:
            np.save(path+networkName.replace(" ", "")+"_ConfusionMatrix",cmIso)
        # Class Definition
        isoclasses = {0: "S1, T1",1: "S1, T2",2: "S1, T3",3: "S1, T4",4: "S1, T5",5: "S1, T6",
                      6: "S2, T1",7: "S2, T2",8: "S2, T3",9: "S2, T4",10:"S2, T5",11:"S2, T6",
                      12:"S3, T1",13:"S3, T2",14:"S3, T3",15:"S3, T4",16:"S3, T5",17:"S3, T6",
                      18:"S4, T1",19:"S4, T2",20:"S4, T3",21:"S4, T4",22:"S4, T5",23:"S4, T6",
                      24:"S5, T1",25:"S5, T2",26:"S5, T3",27:"S5, T4",28:"S5, T5",29:"S5, T6",
                      30:"S6, t1",31:"S6, T2",32:"S6, T3",33:"S6, T4",34:"S6, T5",35:"S6, T6",}
    
    
    
        #%% Plot Comfusion Matrix
        plot_confusion_matrix(cmIso,trueCombinedIso,outputCombinedIso,isoclasses,normalize = "row",title = networkName + " Isolation Confusion Matrix, row normalized",textsize = 12)
        if save:
            plt.savefig(figurePath + networkName.replace(" ", "")+"_FaultIsolation_RowConfusionMatrix.pdf")
        if not networkSelect == "NaiveTest":
            plot_confusion_matrix(cmIso,trueCombinedIso,outputCombinedIso,isoclasses,normalize = "col",title = networkName + " Isolation Confusion Matrix, column normalized",textsize = 12)
            if save:
                plt.savefig(figurePath + networkName.replace(" ", "")+"_FaultIsolation_ColConfusionMatrix.pdf")
        else:
            mostCommon = Counter(outputCombinedIso).most_common(1)[0][0]
            isoMaxList.append(mostCommon)
        #%% Check isolation capability in relation to satellite
        isoSatClasses = {0:"Sat 1", 1:"Sat 2", 2:"Sat 3", 3:"Sat 4", 4:"Sat 5", 5:"Sat 6"}
        cmIsoSat = confusion_matrix(trueCombinedIso//6,outputCombinedIso//6)
        plot_confusion_matrix(cmIsoSat,trueCombinedIso//6,outputCombinedIso//6,
                              isoSatClasses,normalize = "row",
                              title = networkName + " Isolation Confusion Matrix",
                              textsize = 30,colorbar = True)
        if saveCM:
            np.save(path+networkName.replace(" ", "")+"_SatelliteConfusionMatrix",cmIsoSat)
        if save:
            plt.savefig(figurePath+networkName.replace(" ", "")+"_SatelliteConfusionMatrix.pdf", bbox_inches = 'tight')
        
        if not dataset0test:
            isoCorrect = outputCombinedIso == trueCombinedIso
            acc = sum(isoCorrect)/len(isoCorrect)
            accList.append(acc)
            print(networkName + " Isolation Accuracy:",np.round(100* acc,2))
            
            # Isolation Recall (row normalized confusion matrix)
            cmRow = cmIso.astype('float') / cmIso.sum(axis=1)[:, np.newaxis]
            isoRec = np.nanmean(np.diag(cmRow))
            print(networkName + " Isolation Recall: ", np.round(100*isoRec,2))  
            recList.append(isoRec)
            
            # Isolation Precision (column normalized confusion matrix)
            cmCol = cmIso.astype('float') / cmIso.sum(axis=0)[np.newaxis,:]
            isoPrec = np.nanmean(np.diag(cmCol))
            print(networkName + " Isolation Precision: ", np.round(100*isoPrec,2))
            precList.append(isoPrec)
            
            # Isolation Accuracy for closed faults 
            outputCombinedIsoClosed = np.concatenate(isoData[boolClosedFault]['outputVector'].to_numpy())
            trueCombinedIsoClosed = np.concatenate(isoData[boolClosedFault]['trueVector'].to_numpy())
            cmClosed = confusion_matrix(trueCombinedIsoClosed,outputCombinedIsoClosed)
            cmRow = cmClosed.astype('float') / cmClosed.sum(axis=1)[:, np.newaxis]
            isoClosedRec = np.nanmean(np.diag(cmRow))
            
            cmCol = cmClosed.astype('float') / cmClosed.sum(axis=0)[np.newaxis,:]
            isoClosedPrec = np.nanmean(np.diag(cmCol))
            
            isoCorrectClosed = outputCombinedIsoClosed == trueCombinedIsoClosed
            closedAcc = sum(isoCorrectClosed)/len(isoCorrectClosed)   
            print(networkName + " Isolation Accuracy Closed Fault: ", 100*closedAcc)
            print(networkName + " Isolation Recall Closed Fault: ",100*isoClosedRec)
            print(networkName + " Isolation Precision Closed Fault: ",100*isoClosedPrec)
            closedMList.append([closedAcc,isoClosedRec,isoClosedPrec])
            # Isolation Accuracy for open faults 
            outputCombinedIsoOpen = np.concatenate(isoData[boolOpenFault]['outputVector'].to_numpy())
            trueCombinedIsoOpen = np.concatenate(isoData[boolOpenFault]['trueVector'].to_numpy())
            isoCorrectOpen = outputCombinedIsoOpen == trueCombinedIsoOpen
            openAcc = sum(isoCorrectOpen)/len(isoCorrectOpen)   
          
            
            cmOpen = confusion_matrix(trueCombinedIsoOpen,outputCombinedIsoOpen)
            cmRow = cmOpen.astype('float') / cmOpen.sum(axis=1)[:, np.newaxis]
            isoOpenRec = np.nanmean(np.diag(cmRow))
            
            cmCol = cmOpen.astype('float') / cmOpen.sum(axis=0)[np.newaxis,:]
            isoOpenPrec = np.nanmean(np.diag(cmCol))
            openMList.append([openAcc,isoOpenRec,isoOpenPrec])
            print(networkName + " Isolation Accuracy Open Fault: ", 100*openAcc)
            print(networkName + " Isolation Recall Open Fault: ",100*isoOpenRec)
            print(networkName + " Isolation Precision Open Fault: ",100*isoOpenPrec)
            
    #%% Isolation accuracy verus intensity
    if not networkSelect == "NaiveTest":
        paraArray = np.arange(0,9)*0.1 + 0.1
        closedAccParaList = []
        openAccParaList = []
        for para in paraArray:
            boolParaLB = isoData['fParam'] > para
            boolParaUB = isoData['fParam'] < (para + 0.1)
            outPara  = np.concatenate(isoData[boolClosedFault & boolParaLB & boolParaUB]['outputVector'].to_numpy())
            truePara = np.concatenate(isoData[boolClosedFault & boolParaLB & boolParaUB]['trueVector'].to_numpy())
            isoCorrectClosed = outPara == truePara
            closedAcc = sum(isoCorrectClosed)/len(isoCorrectClosed)   
            closedAccParaList.append(closedAcc)
            
            outPara  = np.concatenate(isoData[boolOpenFault & boolParaLB & boolParaUB]['outputVector'].to_numpy())
            truePara = np.concatenate(isoData[boolOpenFault & boolParaLB & boolParaUB]['trueVector'].to_numpy())
            isoCorrectOpen = outPara == truePara
            openAcc = sum(isoCorrectOpen)/len(isoCorrectOpen)   
            openAccParaList.append(openAcc)
        plt.figure()
        plt.plot(paraArray,100*np.array(closedAccParaList))
        plt.minorticks_on()
        plt.grid(b = True, which = 'both')
        plt.ylabel("Isolation Accuracy [-]")
        plt.xlabel('Fault Intensity Parameter [%]')
        plt.ylim([0,105])
        plt.title(networkName + " Closed Fault Isolation Accuracy")
        plt.tight_layout()
        if save:
            plt.savefig(figurePath+networkName.replace(" ","") + "_IsoAccVersusParam_Closed.pdf")
        
        plt.figure()
        plt.plot(paraArray,100*np.array(openAccParaList))
        plt.minorticks_on()
        plt.grid(b = True, which = 'both')
        plt.ylim([0,105])
        plt.ylabel("Isolation Accuracy [%]")
        plt.xlabel('Fault Intensity Parameter [-]')
        plt.title(networkName + " Open Fault Isolation Accuracy")
        plt.tight_layout()
        if save:
            plt.savefig(figurePath+networkName.replace(" ","") + "_IsoAccVersusParam_Open.pdf")
        #%%
        isoAccVsParam = np.zeros([len(isoData),4])
        for j in range(len(isoData)):
            isoAccVsParam[j,0] = isoData['fParam'].iloc[j]
            output = isoData['outputVector'].iloc[j]
            true = isoData['trueVector'].iloc[j]
            isoCorrect = output == true
            isoAccVsParam[j,1] = sum(isoCorrect)/len(isoCorrect)
            _, lenChanges = split_equal_value(output)
            isoAccVsParam[j,2] = len(lenChanges)
            isoAccVsParam[j,3] = np.max(lenChanges)/len(output)
        
        isoAccVsParam = pd.DataFrame(isoAccVsParam, columns = ['fParam','isoAcc','amountChanges','longestSegment'])
        #%%
        paraArray = isoAccVsParam['fParam'].to_numpy()
        isoAccArray = isoAccVsParam['isoAcc'].to_numpy()
        amountArray = isoAccVsParam['amountChanges'].to_numpy()
        
        sortIndex = np.argsort(paraArray)
        isoAccArray = isoAccArray[sortIndex]
        paraArray = paraArray[sortIndex]
         
        nMean = 50
        paraArrayMean = running_mean(paraArray,nMean)
        isoAccArrayMean = running_mean(isoAccArray,nMean)
    
        # Closed Faults
        paraArrayClosed = isoAccVsParam[boolClosedFault]['fParam'].to_numpy()
        isoAccArrayClosed = isoAccVsParam[boolClosedFault]['isoAcc'].to_numpy()
        amountArrayClosed = isoAccVsParam[boolClosedFault]['amountChanges'].to_numpy()
        longestArrayClosed = isoAccVsParam[boolClosedFault]['longestSegment'].to_numpy()
        
        sortIndexClosed = np.argsort(paraArrayClosed)
        isoAccArrayClosed = isoAccArrayClosed[sortIndexClosed]
        paraArrayClosed = paraArrayClosed[sortIndexClosed]
        amountArrayClosed = amountArrayClosed[sortIndexClosed]
        longestArrayClosed = longestArrayClosed[sortIndexClosed]
    
        paraArrayMeanClosed = running_mean(paraArrayClosed,nMean)
        isoAccArrayMeanClosed = running_mean(isoAccArrayClosed,nMean)
        amountArrayMeanClosed = running_mean(amountArrayClosed,nMean)   
        longestArrayMeanClosed = running_mean(longestArrayClosed,nMean)   
        plt.figure(figsize = [1.5*6.4,1.25*4.8])
        plt.plot(paraArrayClosed,amountArrayClosed,'.')
        plt.plot(paraArrayMeanClosed,amountArrayMeanClosed)
        plt.minorticks_on()
        plt.grid(b = True, which = 'both')
        plt.ylabel("Amount of Changes [-]")
        plt.xlabel("Fault Intensity Parameter [-]")
        plt.title(networkName + " Closed Fault Amount of Changes in Isolation")
        plt.tight_layout()
        if save:
            plt.savefig(figurePath+networkName.replace(" ","") + "_AmountOfChanges.pdf")
        
        plt.figure(figsize = [1.5*6.4,1.25*4.8])
        plt.plot(paraArrayClosed,longestArrayClosed,'.')
        plt.plot(paraArrayMeanClosed,longestArrayMeanClosed)
        plt.minorticks_on()
        plt.grid(b = True, which = 'both')
        plt.ylabel("Length [-]")
        plt.xlabel("Fault Intensity Parameter [-]")
        plt.title(networkName + " Closed Fault Longest Continuous Isolation")
        plt.tight_layout()
        
        plt.figure(figsize = [1.5*6.4,1.25*4.8]);
        plt.plot(100*longestArrayClosed,100*isoAccArrayClosed,'.')
        plt.minorticks_on()
        plt.grid(b = True, which = 'both')
        plt.ylabel("Isolation accuracy [%]")
        plt.xlabel("Length of longest continuous isolation w.r.t. total isolation time[%]")
        plt.title(networkName + ": Isolation Accuracy vs longest continuous isolation \n normalized to total isolation time") 
        plt.tight_layout()
        if save:
            plt.savefig(figurePath+networkName.replace(" ","") + "_LengthvsAccuracy.pdf")
        
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
        plt.ylim([0,102])
        # plt.xscale('log')
        # plt.xlim(1e-2,1e-1)
        plt.minorticks_on()
        plt.grid(b = True, which = 'both')
        plt.ylabel("Isolation Accuracy [%]")
        plt.xlabel("Fault Intensity Parameter [-]")
        plt.title(networkName + " Isolation Accuracy")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(figurePath + networkName.replace(" ", "")+"IsoAccBoth_FaultIntensity.pdf")


        #%% Check Time Behavior of the Isolation
        if indRun:
            run = -1
            
            sat = 3
            thr = 3
            fty = 2
            minPara = 0.1
            maxPara = 0.2
            boolSat = isoData['fSat'] == sat
            boolThr = isoData['fThr'] == thr
            boolF = isoData['fType'] == fty
            boolPara = (isoData['fParam'] >= minPara) & (isoData['fParam']<maxPara)
            boolSelect = boolSat & boolThr & boolF & boolPara
            
            if run < 0:
                run = isoData[boolSelect].index[0]
            
            netOutput = isoData['outputVector'].iloc[run]
            trueState = isoData['trueVector'].iloc[run]
            faultType = isoData.iloc[run]['fType']
            fSat = isoData.iloc[run]['fSat']
            fThr = isoData.iloc[run]['fThr']
            fParam = isoData.iloc[run]['fParam']
            
            fileNameRun = isoData.iloc[run]['fName']
            if faultType ==1:
                faultString = "Closed Fault" if faultType == 1 else "Open Fault"
                titleString = faultString + " in Satellite " + str(fSat) + " thruster "+str(fThr) + ", intensity {:.2f}".format(fParam) 
            elif faultType == 2:
                faultString = "Closed Fault" if faultType == 1 else "Open Fault"
                titleString = faultString + " in Satellite " + str(fSat) + " thruster "+str(fThr) + ", intensity {:.2f}".format(fParam) 
            else:
                faultString = "Faultless"
                titleString = "Faultless Case"

            fig,ax  = plt.subplots(figsize=[1.1*6.4, 1.95*4.8])
            ax.plot(trueState + 1,'b')
            ax.plot(netOutput + 1,'.',color ='red')

            ax.set_yticks(np.arange(36)+1)
            ax.set_yticklabels(["S{0}, T{1}".format(i//6+1,i-6*(i//6) + 1) for i in np.arange(36)])
            ax.set(ylabel  = "Selected fault location [-]",
                   ylim = [0,37],
                   xlabel = "Time [s]",
                   title = titleString + "")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(True, which='both',axis = 'x')
            ax.grid(True, which='major',axis = 'y')
            plt.tight_layout()
            if save:
                plt.savefig(figurePath +networkName.replace(" ", "")+"_Isolation_"+faultString.replace(" ", "")+"_"+ str(run) +".pdf")
            saveData = True
            if saveData:
                aniPath = "D:\\Files\\TUDelftLocal\\Thesis\\Greenlight\\Animations\\"
                dataArray = np.vstack( (trueState,netOutput) )
                aniName = networkName.replace(" ", "") + faultString.replace(" ", "")+"_"+ str(run)
                np.savetxt(aniPath + aniName + ".csv", dataArray, delimiter=",")
                
            plt.show()
            _,lenChanges = split_equal_value(netOutput)
            print("Amount of Changes: {}".format(len(lenChanges)))



    #%%
    # paramsClosed = isoData[boolClosedFault]['fParam'].to_numpy()
    # paramsOpen = isoData[boolOpenFault]['fParam'].to_numpy()
    # plt.figure()
    # plt.hist(paramsClosed)
    # plt.figure()
    # plt.hist(paramsOpen)
    # plt.show()
accList = np.array(accList)
precList = np.array(precList)
recList = np.array(recList)
closedMList = np.array(closedMList)
openMList = np.array(openMList)
if networkSelect =="All":
    print("Naive Network")
    print(np.round(100*closedMList[0,0],decimals = 2),"     &",
          np.round(100*closedMList[0,2],decimals = 2),"     &",
          np.round(100*closedMList[0,1],decimals = 2))
    print(np.round(100*openMList[0,0],decimals = 2),"     &",
          np.round(100*openMList[0,2],decimals = 2),"     &",
          np.round(100*openMList[0,1],decimals = 2))
    print(np.round(100*accList[0],decimals = 2),"     &",
          np.round(100*precList[0],decimals = 2),"     &",
          np.round(100*recList[0],decimals = 2))
    ""
    print("Individual Mean")
    print(np.round(100*np.mean(closedMList[1:-1,0]),decimals = 2),"         &",
          np.round(100*np.mean(closedMList[1:-1,2]),decimals = 2),"          &",
          np.round(100*np.mean(closedMList[1:-1,1]),decimals = 2))
    print(np.round(100*np.mean(openMList[1:-1,0]),decimals = 2),"         &",
          np.round(100*np.mean(openMList[1:-1,2]),decimals = 2),"          &",
          np.round(100*np.mean(openMList[1:-1,1]),decimals = 2))
    print(np.round(100*np.mean(accList[1:-1]),decimals = 2),"         &",
          np.round(100*np.mean(precList[1:-1]),decimals = 2),"          &",
          np.round(100*np.mean(recList[1:-1]),decimals = 2))
    
    print("Combined")
    print(np.round(100*closedMList[-1,0],decimals = 2),"    &",
          np.round(100*closedMList[-1,2],decimals = 2),"      &",
          np.round(100*closedMList[-1,1],decimals = 2))
    print(np.round(100*openMList[-1,0],decimals = 2),"    &",
          np.round(100*openMList[-1,2],decimals = 2),"      &",
          np.round(100*openMList[-1,1],decimals = 2))
    print(np.round(100*accList[-1],decimals = 2),"    &",
          np.round(100*precList[-1],decimals = 2),"      &",
          np.round(100*recList[-1],decimals = 2))
elif networkSelect == "NaiveTest":
    fig,ax  = plt.subplots()
    plotArray = list(range(3)) + list(range(4,4+10))
    ax.plot(plotArray,isoMaxList,'.',markersize = 12)
    ax.grid(True, which='both')
    ax.set_yticks(np.unique(isoMaxList))
    ax.set_yticklabels(["S{0}, T{1}".format(i//6+1,i-6*(i//6) + 1) for i in np.unique(isoMaxList)])
    ax.set_xticks(range(max(plotArray)+1))
    ax.set_xticklabels(list(epochArray[:3]) + ["[...]"] + list(epochArray[3:]))
    ax.set(label  = "Most selected fault location [-]",
           xlabel = "Training Epoch [-]",
           title  = "Selected Fault Location of the Naive Network")
