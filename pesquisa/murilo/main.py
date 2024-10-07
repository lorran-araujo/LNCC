
import csv
import numpy
import time
import selector as slctr

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import mkl
mkl.set_num_threads(1)



def cvtHours(t):
    mon, sec = divmod(t, 60)
    hr, mon = divmod(mon, 60)
    return str("%d:%02d:%02d" % (hr, mon, sec))

# Select optimizers
PSO=True
MVO= True
GWO = True
CS= False
BAT=False
DE=False
GA=False
FFA=False
BBO=True
BP=False
BPMA=False
Adam=False

TanSig=True
LogSig=True
ReLU=True

actv_func=[TanSig, LogSig, ReLU]
actv_func_name=["TanSig", "LogSig", "ReLU"]
loss_func=["4"]
loss_func_name=["MSE", "Acuracia", "F1-Score", "Cross-Entropy"]
optimizer=[PSO, MVO, GWO, CS, BAT, DE, GA, FFA, BBO, BP, BPMA, Adam]
optimizer_name=["PSO", "MVO", "GWO", "CS", "BAT", "DE", "GA", "FFA", "BBO", "BP", "BPMA", "Adam"]
datasets=["cancer", "cervical", "diabetes", "diabetes2", "heart", "liver", "parkinsons", "transfusion", "vertebral"]
population_sizes=[10]
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=3

# Select general parameters for all optimizers (population size, number of iterations)
Iterations= 10


runOpt1 = (len(optimizer)-3)*len(loss_func)
runOpt2 = (len(loss_func))*3

#totalRuns = (runOpt1 + runOpt2)*len(actv_func)*len(datasets)*len(population_sizes)*NumOfRuns
totalRuns=len(actv_func)*len(datasets)*len(population_sizes)*NumOfRuns
elapsedRuns=0

timeValues = [[0.253845088,	0.508679796, 1.268109677, 2.561113504],
              [0.04388362, 0.088086227,	0.223086761, 0.465476225],
              [0.257225109,	0.511014648, 1.271087879, 2.506788808],
              [0.19551914, 0.386602648,	0.973274362, 1.918869742],
              [0.117466341,	0.234820487	,0.591162531, 1.156903809],
              [0.117466341,	0.234820487	,0.591162531, 1.156903809],
              [0.125391707,	0.250205979, 0.62441927, 1.216178608],
              [0.098743486,	0.200610573, 0.525546208, 1.035090447],
              [0.267644676,	0.540550699, 1.300523008, 2.489773042],
              [0.255133036,	0.518413583, 1.307878772, 2.569911284]
              ]

totalTime= sum([runOpt1*Iterations*len(actv_func)*NumOfRuns*timeValues[i][j] for i in range(0, len(timeValues)) for j in range(2,len(timeValues[i])-1)])
#print(cvtHours(totalTime))
#input()
#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated file name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))

trainDataset="breastTrain.csv"
testDataset="breastTest.csv"
timeStart=time.time()
for j in range (0, len(datasets)):        # specfiy the number of the datasets
    dataPath = "datasets/"+datasets[j]+".csv"
    data=numpy.loadtxt(open(dataPath,"rb"),delimiter=",",skiprows=0)
    scaler = MinMaxScaler() 
    data = scaler.fit_transform(data)
    k = 0
    kf = KFold(n_splits=NumOfRuns, shuffle=True)
    for train, test in kf.split(data):
        k = k+1
        trainDataset, testDataset = data[train], data[test]
        for i in range (0, len(optimizer)):
        
            if((optimizer[i]==True)): # start experiment if an optimizer and an objective function is selected
                for ai in range(0, len(actv_func)):
                    if(actv_func[ai]==True):
                        for b in range(0,len(loss_func)):
                            func_details=["costNN"+loss_func[b],-1,1]
                            #if(i > 7 and b < 2):
                            #    continue
                            for p in population_sizes:
                                x=slctr.selector(i,func_details,p,Iterations,trainDataset,testDataset, ai)
                                elapsedRuns +=1
                                timeEnd = time.time()
                                timeElapsed=timeEnd-timeStart
                                timeRemaining=totalTime-timeElapsed
                                print(str(numpy.round((elapsedRuns/totalRuns*100),2)) + "%/100%" + "\tTime Elapsed:"+str(cvtHours(timeElapsed))+"\tTime Remaining:"+str(cvtHours(timeRemaining)))                                     
                                if(Export==True):
                                    with open(ExportToFile, 'a',newline='\n') as out:
                                        writer = csv.writer(out,delimiter=',')
                                        if (Flag==False): # just one time to write the header of the CSV file
                                            header= numpy.concatenate([["Optimizer","Population","Dataset","objfname","K run","Activation Function","Loss Function","ExecutionTime", "IterationTime","trainAcc", "trainTP","trainFN","trainFP","trainTN", "testAcc", "testTP","testFN","testFP","testTN"],CnvgHeader])
                                            writer.writerow(header)
                                        if(x.executionTime != -1):
                                            a=numpy.concatenate([[x.optimizer,p,datasets[j],x.objfname,k,actv_func_name[ai],loss_func_name[b],x.executionTime,x.iterationTime,x.trainAcc, x.trainTP,x.trainFN,x.trainFP,x.trainTN, x.testAcc, x.testTP,x.testFN,x.testFP,x.testTN],x.convergence])
                                            writer.writerow(a)
                                    out.close()
                                Flag=True # at least one experiment
                                if(i > 7):
                                    break
if (Flag==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        
