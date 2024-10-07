# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:44:52 2016

@author: hossam
"""
import Adam as adam
import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import CS as cs
import BAT as bat
import BBO as bbo
import DE as de
import GA as ga
import FFA as fa
import csv
import numpy # type: ignore
import time
import neurolab as nl
import costNN
import evaluateNetClassifier as evalNet
from solution import solution
from sklearn.metrics import confusion_matrix # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore

def selector(algo,func_details,popSize,Iter,trainDataset,testDataset, actv):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    
    
    Dataset_train=trainDataset
    Dataset_test=testDataset
      
      
    
    numRowsTrain=numpy.shape(Dataset_train)[0]    # number of instances in the train dataset
    numInputsTrain=numpy.shape(Dataset_train)[1]-1 #number of features in the train dataset

    numRowsTest=numpy.shape(Dataset_test)[0]    # number of instances in the test dataset
    
    numInputsTest=numpy.shape(Dataset_test)[1]-1 #number of features in the test dataset
 

    trainInput=Dataset_train[0:numRowsTrain,0:-1]
    trainOutput=Dataset_train[0:numRowsTrain,-1]
    
    testInput=Dataset_test[0:numRowsTest,0:-1]
    testOutput=Dataset_test[0:numRowsTest,-1]
    
    #number of hidden neurons
    HiddenNeurons = numInputsTrain*2+1
    net = nl.net.newff([[0, 1]]*numInputsTrain, [HiddenNeurons, 1])
    if(actv==1):
        net = nl.net.newff([[0, 1]]*numInputsTrain, [HiddenNeurons, 1], [nl.trans.LogSig(), nl.trans.LogSig()])
    if(actv==2):
        net = nl.net.newff([[0, 1]]*numInputsTrain, [HiddenNeurons, 1], [nl.trans.SatLinPrm(1, 0, 1), nl.trans.SatLinPrm(1, 0, 1)])
    
    dim=(numInputsTrain*HiddenNeurons)+(2*HiddenNeurons)+1;
    if(algo==12):
        x=adam.adamse( getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==0):
        x=pso.PSO( getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==1):
        x=mvo.MVO(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==2):
        x=gwo.GWO( getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==3):
        x=cs.CS(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==4):
        x=bat.BAT(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==5):
        x=de.DE(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==6):
        x=ga.GA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==7):
        x=fa.FFA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==8):
        x=bbo.BBO(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==9):
        
        printAcc=[]
        printAcc2=[]
        
        x = solution()
        timerStart=time.time() 
        x.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        if(function_name=="costNN"):    
            net.trainf.defaults['trainf'] = nl.error.MSE()
        elif(function_name=="costNN4"):
            net.trainf.defaults['trainf'] = nl.error.CEE()
        else:
            return x
        net.trainf = nl.train.train_gd
        newOutput = [[x] for x in trainOutput]
        newOutput = numpy.asarray(newOutput)
        e = net.train(trainInput, newOutput, epochs=Iter*popSize)
        timerEnd=time.time()  
        x.optimizer="BP"
        x.objfname=function_name
        x.popnum=0
        x.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        x.executionTime=timerEnd-timerStart
        x.convergence = e
        pred=net.sim(trainInput).reshape(len(trainOutput))
        pred=numpy.round(pred).astype(int)   
        trainOutput=trainOutput.astype(int) 
        pred=numpy.clip(pred, 0, 1)
        ConfMatrix=confusion_matrix(trainOutput, pred)     
        ConfMatrix1D=ConfMatrix.flatten()
        printAcc.append(accuracy_score(trainOutput, pred,normalize=True)) 
        classification_results= numpy.concatenate((printAcc,ConfMatrix1D))
        x.trainAcc=classification_results[0]
        x.trainTP=classification_results[1]
        x.trainFN=classification_results[2]
        x.trainFP=classification_results[3]
        x.trainTN=classification_results[4]
        
        pred=net.sim(testInput).reshape(len(testOutput))
        pred=numpy.round(pred).astype(int)   
        testOutput=testOutput.astype(int) 
        pred=numpy.clip(pred, 0, 1)
        ConfMatrix=confusion_matrix(testOutput, pred)     
        ConfMatrix1D=ConfMatrix.flatten()
        printAcc2.append(accuracy_score(testOutput, pred,normalize=True)) 
        classification_results2= numpy.concatenate((printAcc2,ConfMatrix1D))
        x.testAcc=classification_results2[0]
        x.testTP=classification_results2[1]
        x.testFN=classification_results2[2]
        x.testFP=classification_results2[3]
        x.testTN=classification_results2[4]
        
        return x
    if(algo==10):
        
        printAcc=[]
        printAcc2=[]
        x = solution()
        timerStart=time.time() 
        x.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        if(function_name=="costNN"):    
            net.trainf.defaults['trainf'] = nl.error.MSE()
        elif(function_name=="costNN4"):
            net.trainf.defaults['trainf'] = nl.error.CEE()
        else:
            return x
        net.trainf = nl.train.train_gdx
        newOutput = [[x] for x in trainOutput]
        newOutput = numpy.asarray(newOutput)
        e = net.train(trainInput, newOutput, epochs=Iter*popSize)
        timerEnd=time.time()  
        x.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        x.optimizer="BPMA"
        x.objfname=function_name
        x.popnum=0
        x.executionTime=timerEnd-timerStart
        x.convergence = e
        pred=net.sim(trainInput).reshape(len(trainOutput))
        pred=numpy.round(pred).astype(int)   
        trainOutput=trainOutput.astype(int) 
        pred=numpy.clip(pred, 0, 1)
        ConfMatrix=confusion_matrix(trainOutput, pred)     
        ConfMatrix1D=ConfMatrix.flatten()
        printAcc.append(accuracy_score(trainOutput, pred,normalize=True)) 
        classification_results= numpy.concatenate((printAcc,ConfMatrix1D))
        x.trainAcc=classification_results[0]
        x.trainTP=classification_results[1]
        x.trainFN=classification_results[2]
        x.trainFP=classification_results[3]
        x.trainTN=classification_results[4]
        
        pred=net.sim(testInput).reshape(len(testOutput))
        pred=numpy.round(pred).astype(int)   
        testOutput=testOutput.astype(int) 
        pred=numpy.clip(pred, 0, 1)
        ConfMatrix=confusion_matrix(testOutput, pred)     
        ConfMatrix1D=ConfMatrix.flatten()
        printAcc2.append(accuracy_score(testOutput, pred,normalize=True)) 
        classification_results2= numpy.concatenate((printAcc2,ConfMatrix1D))
        x.testAcc=classification_results2[0]
        x.testTP=classification_results2[1]
        x.testFN=classification_results2[2]
        x.testFP=classification_results2[3]
        x.testTN=classification_results2[4]
        
        return x
    if(algo==11):
        x = solution()
        printAcc=[]
        printAcc2=[]
        if(function_name=="costNN4"):
            if(actv==0):
                timerStart=time.time() 
                x.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
                clf = MLPClassifier(hidden_layer_sizes=HiddenNeurons, activation='tanh', max_iter=Iter*popSize, learning_rate_init=0.01, n_iter_no_change=7500).fit(trainInput,trainOutput)
               
                timerEnd=time.time()  
                x.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
                x.optimizer="Adam"
                x.objfname=function_name
                x.popnum=0
                x.executionTime=timerEnd-timerStart
                x.convergence = clf.loss_curve_
                
                
                pred = clf.predict(trainInput)
                ConfMatrix=confusion_matrix(trainOutput, pred)     
                ConfMatrix1D=ConfMatrix.flatten()
                printAcc.append(accuracy_score(trainOutput, pred,normalize=True)) 
                classification_results= numpy.concatenate((printAcc,ConfMatrix1D))
                x.trainAcc=classification_results[0]
                x.trainTP=classification_results[1]
                x.trainFN=classification_results[2]
                x.trainFP=classification_results[3]
                x.trainTN=classification_results[4]
                
                pred = clf.predict(testInput)
                ConfMatrix=confusion_matrix(testOutput, pred)     
                ConfMatrix1D=ConfMatrix.flatten()
                printAcc2.append(accuracy_score(testOutput, pred,normalize=True)) 
                classification_results2= numpy.concatenate((printAcc2,ConfMatrix1D))
                x.testAcc=classification_results2[0]
                x.testTP=classification_results2[1]
                x.testFN=classification_results2[2]
                x.testFP=classification_results2[3]
                x.testTN=classification_results2[4]
                
                return x
            elif(actv==1):
                timerStart=time.time() 
                x.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
                clf = MLPClassifier(hidden_layer_sizes=HiddenNeurons, activation='logistic', max_iter=Iter*popSize, learning_rate_init=0.01, n_iter_no_change=7500).fit(trainInput,trainOutput)
                timerEnd=time.time()  
                x.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
                x.optimizer="Adam"
                x.objfname=function_name
                x.popnum=0
                x.executionTime=timerEnd-timerStart
                x.convergence = clf.loss_curve_
                
                pred = clf.predict(trainInput)
                ConfMatrix=confusion_matrix(trainOutput, pred)     
                ConfMatrix1D=ConfMatrix.flatten()
                printAcc.append(accuracy_score(trainOutput, pred,normalize=True)) 
                classification_results= numpy.concatenate((printAcc,ConfMatrix1D))
                x.trainAcc=classification_results[0]
                x.trainTP=classification_results[1]
                x.trainFN=classification_results[2]
                x.trainFP=classification_results[3]
                x.trainTN=classification_results[4]
                
                pred = clf.predict(testInput)
                ConfMatrix=confusion_matrix(testOutput, pred)     
                ConfMatrix1D=ConfMatrix.flatten()
                printAcc2.append(accuracy_score(testOutput, pred,normalize=True)) 
                classification_results2= numpy.concatenate((printAcc2,ConfMatrix1D))
                x.testAcc=classification_results2[0]
                x.testTP=classification_results2[1]
                x.testFN=classification_results2[2]
                x.testFP=classification_results2[3]
                x.testTN=classification_results2[4]
                
                return x
            elif(actv==2):
                timerStart=time.time() 
                x.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
                clf = MLPClassifier(hidden_layer_sizes=HiddenNeurons, activation='relu', max_iter=Iter*popSize, learning_rate_init=0.01, n_iter_no_change=7500).fit(trainInput,trainOutput)
                timerEnd=time.time()  
                x.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
                x.optimizer="Adam"
                x.objfname=function_name
                x.popnum=0
                x.executionTime=timerEnd-timerStart
                x.convergence = clf.loss_curve_
                
                pred = clf.predict(trainInput)
                ConfMatrix=confusion_matrix(trainOutput, pred)     
                ConfMatrix1D=ConfMatrix.flatten()
                printAcc.append(accuracy_score(trainOutput, pred,normalize=True)) 
                classification_results= numpy.concatenate((printAcc,ConfMatrix1D))
                x.trainAcc=classification_results[0]
                x.trainTP=classification_results[1]
                x.trainFN=classification_results[2]
                x.trainFP=classification_results[3]
                x.trainTN=classification_results[4]
                
                pred = clf.predict(testInput)
                ConfMatrix=confusion_matrix(testOutput, pred)     
                ConfMatrix1D=ConfMatrix.flatten()
                printAcc2.append(accuracy_score(testOutput, pred,normalize=True)) 
                classification_results2= numpy.concatenate((printAcc2,ConfMatrix1D))
                x.testAcc=classification_results2[0]
                x.testTP=classification_results2[1]
                x.testFN=classification_results2[2]
                x.testFP=classification_results2[3]
                x.testTN=classification_results2[4]
                
                return x
        else:
            return x
            
    # Evaluate MLP classification model based on the training set
    trainClassification_results=evalNet.evaluateNetClassifier(x,trainInput,trainOutput,net)
    x.trainAcc=trainClassification_results[0]
    x.trainTP=trainClassification_results[1]
    x.trainFN=trainClassification_results[2]
    x.trainFP=trainClassification_results[3]
    x.trainTN=trainClassification_results[4]
   
    # Evaluate MLP classification model based on the testing set   
    testClassification_results=evalNet.evaluateNetClassifier(x,testInput,testOutput,net)
    x.testAcc=testClassification_results[0]
    x.testTP=testClassification_results[1]
    x.testFN=testClassification_results[2]
    x.testFP=testClassification_results[3]
    x.testTN=testClassification_results[4] 
    
    
    return x
    
#####################################################################    
