# gradient descent optimization with adam for a two-dimensional test function
from math import sqrt
import numpy as np
from numpy.random import rand
from numpy.random import seed
import random
import math
from solution import solution
import time
 
# objective function
def objective(x, inputs,outputs,net):
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
 
# derivative of objective function
#def derivative(x, y):
def derivative(x, inputs,outputs,net):
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
    #mse = -2*((pred - trainOutput)).mean(axis=None)
    
    
    return mse
     
# gradient descent algorithm with adam
def adam(objf,lb,ub,dim,PopSize,iters,trainInput,trainOutput,net):
     alpha = 0.01
     # factor for average gradient
     beta1 = 0.9
     # factor for average squared gradient
     beta2 = 0.999
     eps=1e-8
     
     
     s=solution()
     # generate an initial point
     pos=np.random.uniform(0,1,(dim)) *(ub-lb)+lb
     score = objective(pos, trainInput,trainOutput,net)
     
     convergence_curve=np.zeros((PopSize*iters)+1)
    
     ############################################
     timerStart=time.time() 
     s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
     
     # initialize first and second moments
     m = np.zeros(dim)
     v = np.zeros(dim)
     #print(m.shape)
     #print(v.shape)
     # run the gradient descent updates
     for t in range(iters*PopSize):
         # calculate gradient g(t)
         g = derivative(pos, trainInput,trainOutput,net)
         #print(g.shape)
         # build a solution one variable at a time
         for i in range(pos.shape[0]):
             # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
             m[i] = beta1 * m[i] + (1.0 - beta1) * g
             # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
             v[i] = beta2 * v[i] + (1.0 - beta2) * g**2
             # mhat(t) = m(t) / (1 - beta1(t))
             mhat = m[i] / (1.0 - beta1**(t+1))
             # vhat(t) = v(t) / (1 - beta2(t))
             vhat = v[i] / (1.0 - beta2**(t+1))
             # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
             pos[i] = pos[i] - alpha * mhat / (sqrt(vhat) + eps)
          # evaluate candidate point
         score = objective(pos, trainInput,trainOutput,net)
         convergence_curve[t]=score
         #print('>%d f(%s) = %.5f' % (t, pos, score))
        # report progress
             #
             #return [pos, score]
     
     timerEnd=time.time()  
     s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
     s.iterationTime = s.iterationTime/iters
     s.executionTime=timerEnd-timerStart
     s.convergence=convergence_curve
     s.optimizer="Adam"
     s.objfname=objf.__name__
     s.bestIndividual=pos

     return s