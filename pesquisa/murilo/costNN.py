# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:03:15 2016

@author: hossam
"""
import numpy as np
import neurolab as nl
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

def costNN(x,inputs,outputs,net):
    trainInput=inputs
    
    
    trainOutput=outputs
    
    numInputs=np.shape(trainInput)[1] #number of inputs
    
    
    #number of hidden neurons
    HiddenNeurons = net.layers[0].np['b'][:].shape[0]
   
    ######################################

    
    split1=HiddenNeurons*numInputs
    split2=split1+HiddenNeurons
    split3=split2+HiddenNeurons
    
    # input_w = 3X8 (HiddenNeurons*numInputs) 
    input_w =x[0:split1].reshape(HiddenNeurons,numInputs)
                       
    # layer_w = 1 X 3 (HiddenNeurons)
    layer_w=x[split1:split2].reshape(1,HiddenNeurons)
 
    # input_bias = hiddenNeurons
    input_bias=x[split2:split3].reshape(1,HiddenNeurons)
    #input_bias = np.array([0.4747,-1.2475,-1.2470])
    
    # bias_2 = 1
    bias_2 =x[split3:split3+1]

    
    net.layers[0].np['w'][:] = input_w
    net.layers[1].np['w'][:] = layer_w
    net.layers[0].np['b'][:] = input_bias
    net.layers[1].np['b'][:] = bias_2
    
    
    
    pred=net.sim(trainInput).reshape(len(trainOutput))
    
    mse = ((pred - trainOutput) ** 2).mean(axis=None)
    
    
    
    return mse

def costNN2(x,inputs,outputs,net):
    trainInput=inputs
    
    
    trainOutput=outputs
    
    numInputs=np.shape(trainInput)[1] #number of inputs
    
    
    #number of hidden neurons
    HiddenNeurons = net.layers[0].np['b'][:].shape[0]
   
    ######################################

    
    split1=HiddenNeurons*numInputs
    split2=split1+HiddenNeurons
    split3=split2+HiddenNeurons
    
    # input_w = 3X8 (HiddenNeurons*numInputs) 
    input_w =x[0:split1].reshape(HiddenNeurons,numInputs)
                       
    # layer_w = 1 X 3 (HiddenNeurons)
    layer_w=x[split1:split2].reshape(1,HiddenNeurons)
 
    # input_bias = hiddenNeurons
    input_bias=x[split2:split3].reshape(1,HiddenNeurons)
    #input_bias = np.array([0.4747,-1.2475,-1.2470])
    
    # bias_2 = 1
    bias_2 =x[split3:split3+1]

    
    net.layers[0].np['w'][:] = input_w
    net.layers[1].np['w'][:] = layer_w
    net.layers[0].np['b'][:] = input_bias
    net.layers[1].np['b'][:] = bias_2
    
    
    
    pred=net.sim(trainInput).reshape(len(trainOutput))
    #scaler = MinMaxScaler()
    #pred = pred.reshape(-1, 1)
    #pred = scaler.fit_transform(pred)
    #pred = pred.flatten()
    pred=np.round(pred).astype(int)   
    trainOutput=trainOutput.astype(int) 
    pred=np.clip(pred, 0, 1)

    acc = accuracy_score(trainOutput, pred,normalize=True)
    acc = 1-acc
    return acc


def costNN3(x,inputs,outputs,net):
    trainInput=inputs
    
    
    trainOutput=outputs
    
    numInputs=np.shape(trainInput)[1] #number of inputs
    
    
    #number of hidden neurons
    HiddenNeurons = net.layers[0].np['b'][:].shape[0]
   
    ######################################

    
    split1=HiddenNeurons*numInputs
    split2=split1+HiddenNeurons
    split3=split2+HiddenNeurons
    
    # input_w = 3X8 (HiddenNeurons*numInputs) 
    input_w =x[0:split1].reshape(HiddenNeurons,numInputs)
                       
    # layer_w = 1 X 3 (HiddenNeurons)
    layer_w=x[split1:split2].reshape(1,HiddenNeurons)
 
    # input_bias = hiddenNeurons
    input_bias=x[split2:split3].reshape(1,HiddenNeurons)
    #input_bias = np.array([0.4747,-1.2475,-1.2470])
    
    # bias_2 = 1
    bias_2 =x[split3:split3+1]

    
    net.layers[0].np['w'][:] = input_w
    net.layers[1].np['w'][:] = layer_w
    net.layers[0].np['b'][:] = input_bias
    net.layers[1].np['b'][:] = bias_2
    
    
    
    pred=net.sim(trainInput).reshape(len(trainOutput))
    #scaler = MinMaxScaler()
    #pred = pred.reshape(-1, 1)
    #pred = scaler.fit_transform(pred)
    #pred = pred.flatten()
    #ConfMatrix=confusion_matrix(trainOutput, pred.round())
    #tn, fp, fn, tp=confusion_matrix(trainOutput, pred.round()).ravel()
    #precision = tp/(tp+fp)
    #recall = tp/(tp+fn)
    #f1 = 2*((precision*recall)/(precision+recall))
    #print(ConfMatrix)
    #print(precision)
    #print(recall)
    #print(f1)
    #print(trainOutput)
    #print(pred.round())
    #print(pred)
    pred=np.round(pred).astype(int)   
    trainOutput=trainOutput.astype(int) 
    pred=np.clip(pred, 0, 1)

    f1 = f1_score(trainOutput, pred)
    f1 = 1-f1
    #print(f1)    
    
    
    return f1

def costNN4(x,inputs,outputs,net):
    trainInput=inputs
    
    
    trainOutput=outputs
    
    numInputs=np.shape(trainInput)[1] #number of inputs
    
    
    #number of hidden neurons
    HiddenNeurons = net.layers[0].np['b'][:].shape[0]
   
    ######################################

    
    split1=HiddenNeurons*numInputs
    split2=split1+HiddenNeurons
    split3=split2+HiddenNeurons
    
    # input_w = 3X8 (HiddenNeurons*numInputs) 
    input_w =x[0:split1].reshape(HiddenNeurons,numInputs)
                       
    # layer_w = 1 X 3 (HiddenNeurons)
    layer_w=x[split1:split2].reshape(1,HiddenNeurons)
 
    # input_bias = hiddenNeurons
    input_bias=x[split2:split3].reshape(1,HiddenNeurons)
    #input_bias = np.array([0.4747,-1.2475,-1.2470])
    
    # bias_2 = 1
    bias_2 =x[split3:split3+1]

    
    net.layers[0].np['w'][:] = input_w
    net.layers[1].np['w'][:] = layer_w
    net.layers[0].np['b'][:] = input_bias
    net.layers[1].np['b'][:] = bias_2
    
    
    
    pred=net.sim(trainInput).reshape(len(trainOutput))
    loss = log_loss(trainOutput, pred)
    
    
    
    return loss 