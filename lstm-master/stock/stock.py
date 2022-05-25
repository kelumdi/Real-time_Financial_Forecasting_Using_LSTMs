
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data
import math
import numpy as np
import pandas as pd
import time
import pickle

np.random.seed(0)
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# get trainInput, trainLabel, testInput, testLabel by window-based-data
def getDataTrainTest_1Label(pathStock, normalize, ss, winInput, numPredic):
    dfStock2 = pd.read_csv(pathStock, header=None).to_numpy()

    if normalize == True:
        dfStock2 = ss.fit_transform(dfStock2)
    else:
        dfStock2 = dfStock2

    # call function to  get trainInput, trainLabel, testInput, testLabel by window-based-data : train 1228days & test 1day
    (trainInputL, trainLabelL, testInputL, testLabelL) = miniBatchOneDayLabel_1Label_5yr(dfStock2, winInput, numPredic)

    return (trainInputL, trainLabelL, testInputL, testLabelL, ss, dfStock2)

# get trainInput, trainLabel, testInput, testLabel by window-based-data : train 1228days & test 1day
def miniBatchOneDayLabel_1Label_5yr(inputt, winInput, numPredic):

    #1.get trainInput, trainLabel
    input2 = inputt.copy()
    input3 = input2.copy()
    indexFirstPrediction = inputt.shape[0] - 30  #
    indexL = np.linspace(0, indexFirstPrediction - winInput + numPredic - 2,
                         num=indexFirstPrediction - winInput + numPredic - 1)

    dataTrainL = []
    labelTrainL = []
    for i in indexL:
        i = int(i)
        ii = input2[i:(i + winInput), :]
        ll = input3[(i + winInput):(i + 1 + winInput), :]
        dataTrainL.append(ii)
        labelTrainL.append(ll)
    input2 = inputt.copy()
    input3 = input2.copy()
    indexL = np.linspace(1, indexFirstPrediction - winInput + numPredic - 1,
                         num=indexFirstPrediction - winInput + numPredic - 1)

    #2. get testInput, testLabel
    dataTestL = []
    labelTestL = []
    for i in indexL:
        i = int(i)
        ii = input2[i:(i + winInput), :]
        ll = input3[(i + winInput):(i + 1 + winInput), :]
        dataTestL.append(ii)
        labelTestL.append(ll)
    return (dataTrainL, labelTrainL, dataTestL, labelTestL)


# https://towardsdatascience.com/lstms-in-pytorch-528b0440244
# https://github.com/wcneill/jn-ml-textbook/blob/master/Deep%20Learning/04%20Recurrent%20Networks/pytorch13b_LSTM.ipynb
# https://github.com/wcneill/jn-ml-textbook/blob/master/Deep%20Learning/04%20Recurrent%20Networks/pytorch13_char_RNN.ipynb
class LSTM3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hs):
        out, hs = self.lstm(x, hs)
        out = self.fc(out)
        return out, hs

# get a pair of input & label
def getData2(data, label):
    y1List = data
    y2List = label
    y1L5 = [torch.Tensor(e) for i, e in enumerate(y1List)]
    y2L5 = [torch.Tensor(e) for i, e in enumerate(y2List)]
    twoList = [(e1, e2) for i, e1 in enumerate(y1L5) for j, e2 in enumerate(y2L5) if i == j] # get a pair of input & label
    return (y1L5, y2L5, twoList)

# call the function to get a pair of input & label
def getData(data, label):
    (y1, y2, y12) = getData2(data, label)
    return (y12)

#start training and testing
def trainLSTM3MSE(normalize, ss, model,n_epochs, lrRnn, train_loader, test_loader, batchSizeTrain,
                  batchSizeTest, seq_lengthInput, seq_lengthLabel, inputSize, outputSize):
    criterion = nn.MSELoss() #meas squared loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lrRnn) #adam optimization for backpropagation

    # Training Run
    epochList = []
    lossListTrain = []
    lossListTest = []
    for epoch in range(1, n_epochs + 1): #start epoch
        start = time.time()
        epochList.append(epoch)
        lossSumTrain = 0
        hs = None
        outputTrainL = []
        labelTrainL=[]
        outputTestL = []
        labelTestL = []
        for batch_idx, (input, label) in enumerate(train_loader): #train batch
            for batch_idx2, (inputTest, labelTest) in enumerate(test_loader): #test batch
                if batch_idx == batch_idx2: #run only when train-batch-index is same as test-batch-index
                    #change [b, inputSize,sequentialLength] to [b, sequentialLength, inputSize]
                    input = input.view(batchSizeTrain, seq_lengthInput,inputSize).float()
                    label = label.view(batchSizeTrain, seq_lengthLabel,outputSize).float()

                    outputTrain, h_state = model(input, hs)
                    outputTrain = outputTrain.float()
                    lossTrain = criterion(outputTrain, label) #apply our loss function with outputTrain and its label
                    optimizer.zero_grad()  # Clears existing gradients from previous epoch
                    lossTrain.backward()  # Does backpropagation and calculates gradients
                    optimizer.step()
                    lossSumTrain = lossSumTrain + lossTrain.detach().numpy() #sum train loss
                    outputTrainL.append(outputTrain) #list outpurTrain
                    labelTrainL.append(label)

                    with torch.no_grad():
                        lossSumTest = 0
                        # change [b, inputSize,sequentialLength] to [b, sequentialLength, inputSize]
                        inputTest = inputTest.view(batchSizeTest, seq_lengthInput, inputSize).float()
                        labelTest = labelTest.view(batchSizeTest, seq_lengthLabel, outputSize).float()
                        # Forward pass only to get logits/output
                        outputTest, h_stateTest = model(inputTest, hs)
                        lossTest = criterion(outputTest, labelTest) #apply our loss function with outputTest and its label
                        lossSumTest = lossSumTest + lossTest.detach().numpy() #sum test loss
                        outputTestL.append(outputTest) #list test output
                        labelTestL.append(labelTest)

        lossTrainAvgForOneEpoch1 = lossSumTrain / len(train_loader.dataset) #train loss for each epoch
        lossListTrain.append(lossTrainAvgForOneEpoch1)

        lossTestAvgForOneEpoch1 = lossSumTest / len(test_loader.dataset) #test loss for each epoch
        lossListTest.append(lossTestAvgForOneEpoch1)
        end = time.time()

        if epoch == 1:
            # 1. Train :
            outputTrainL2 = []
            for i in outputTrainL:
                i = i.detach().numpy()
                ii = i.reshape(i.shape[2], -1)
                outputTrainL2.append(ii) #list train output
            labelTrainL2 = []
            for i in labelTrainL:
                i = i.detach().numpy()
                ii = i.reshape(i.shape[2], -1)
                labelTrainL2.append(ii)  #list train label
            outputTrainL3 = outputTrainL2.copy()
            trainL0 = outputTrainL3[0] # get scaler from a list

            if normalize == True:
                trainL1 = ss.inverse_transform(trainL0) #apply inverse transformation for standardization
            else:
                trainL1 = trainL0

            # 2. Test :
            outputTestL2 = []
            for i in outputTestL:
                i = i.detach().numpy()
                ii = i.reshape(i.shape[2], -1)
                outputTestL2.append(ii) #list test output
            labelTestL2 = []
            for i in labelTestL:
                i = i.detach().numpy()
                ii = i.reshape(i.shape[2], -1)
                labelTestL2.append(ii) #list test label
            outputTestL3 = outputTestL2.copy()
            testL0 = outputTestL3[0] # get scaler from a list

            if normalize == True:
                testL1 = ss.inverse_transform(testL0)
            else:
                testL1 = testL0

            print("Epoch :", epoch, '// One epoch time:', end - start, '// Train Error :', lossTrainAvgForOneEpoch1,
                  '// Test Error :', lossTestAvgForOneEpoch1)

        elif epoch > 1:
            ###############################################
            # find and save the minimum loss for each epoch
            ###############################################
            minLossTrain = min(lossListTrain[0:(epoch - 1)])
            if minLossTrain > lossTrainAvgForOneEpoch1:

                # 1. Train :
                outputTrainL2 = []
                for i in outputTrainL:
                    i = i.detach().numpy()
                    ii = i.reshape(i.shape[2], -1)
                    outputTrainL2.append(ii) #list train output
                labelTrainL2 = []
                for i in labelTrainL:
                    i = i.detach().numpy()
                    ii = i.reshape(i.shape[2], -1)
                    labelTrainL2.append(ii) #list train label
                outputTrainL3 = outputTrainL2.copy()
                trainL0 = outputTrainL3[0] # get scaler from a list
                if normalize == True:
                    trainL1 = ss.inverse_transform(trainL0)
                else:
                    trainL1=trainL0

                # 2. Test :
                outputTestL2 = []
                for i in outputTestL:
                    i = i.detach().numpy()
                    ii = i.reshape(i.shape[2], -1)
                    outputTestL2.append(ii) #list test output
                labelTestL2 = []
                for i in labelTestL:
                    i = i.detach().numpy()
                    ii = i.reshape(i.shape[2], -1)
                    labelTestL2.append(ii) #list test label
                outputTestL3 = outputTestL2.copy()
                testL0 = outputTestL3[0] # get scaler from a list

                if normalize == True:
                    testL1 = ss.inverse_transform(testL0) #apply inverse transformation for standardization
                else:
                    testL1 = testL0

                print("Epoch :", epoch, '// One epoch time:', end - start, '// Train Error :', lossTrainAvgForOneEpoch1,
                      '// Test Error :', lossTestAvgForOneEpoch1)
            else:
                print("Epoch :", epoch, ": Current loss is bigger than previous")
                continue
    return (trainL1,testL1)

#prepare data and start run train and test day by day
def runDayByDay(trainInputL2, trainLabelL2, testInputL2, testLabelL2, shuffleTrain,batchSizeTrain,batchSizeTest,input_size,
                hidden_size, nLayer, output_size, dropout1,normalize, ss, n_epochs, lrRnnAdam,seqLengthInput, seqLengthLabel):
    # get train data (trainInput,trainLabel) and test data (testInput,testLabel)
    train_xy = getData(trainInputL2, trainLabelL2)
    test_xy = getData(testInputL2, testLabelL2)
    # apply train data and test data into dataloader of pytorch
    if shuffleTrain == True:
        train_loader = torch.utils.data.DataLoader(dataset=train_xy,
                                                   batch_size=batchSizeTrain,
                                                   shuffle=True)
    elif shuffleTrain == False:
        train_loader = torch.utils.data.DataLoader(dataset=train_xy,
                                                   batch_size=batchSizeTrain,
                                                   shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_xy,
                                              batch_size=batchSizeTest,
                                              shuffle=False)

    model = LSTM3(input_size, hidden_size, nLayer, output_size, dropout1) #define our model
    # call training and testing
    (predicTrain,predicTest) = trainLSTM3MSE(normalize, ss, model, n_epochs, lrRnnAdam,train_loader, test_loader, batchSizeTrain, batchSizeTest,
                           seqLengthInput, seqLengthLabel, input_size, output_size)
    return(predicTrain, predicTest)

def main():

    winInput = 1228 #input size for one batch
    winLabel = 1 #label size for one batch
    shuffleTrain = False #We do not shuffle

    dropout1 = 0 #did not use dropout regularization
    normalize = True #need to standardize

    numPredic = 30 #We predict 30days
    stock = 'apple' #name of our stock

    lrRnnAdam = 0.01
    n_epochs = 3  #

    # get trainInput, trainLabel,testInput,testLabel for each stock
    if stock == 'MS':
        stock1 = './data/trueMSFT_5yr.csv'
        ss = StandardScaler() #standardize
        (trainInputL, trainLabelL, testInputL, testLabelL, ss,
         dfStock2) = getDataTrainTest_1Label(stock1, normalize, ss, winInput, numPredic)
    elif stock == 'apple':
        stock1 = './data/trueAAPL_5yr.csv'
        ss = StandardScaler()
        (trainInputL, trainLabelL, testInputL, testLabelL, ss,
         dfStock2) = getDataTrainTest_1Label(stock1, normalize, ss, winInput, numPredic)
    elif stock == 'google':
        stock1 = './data/trueGOOGL_5yr.csv'
        ss = StandardScaler()
        (trainInputL, trainLabelL, testInputL, testLabelL, ss,dfStock2) = getDataTrainTest_1Label(stock1, normalize, ss, winInput, numPredic)

    # batch size
    batchSizeTrain = 1
    batchSizeTest = 1

    time_step = winInput

    #set parameter values for LSTM
    input_size = time_step
    seqLengthInput = 1
    seqLengthLabel = 1
    hidden_size = 512  #number of hidden node
    nLayer = 1 #number of LSTM
    output_size = 1

    outWinL = []

    #start training and testing from each test day
    for j in range(len(testInputL)-1):
        trainInputL2 = [trainInputL[j]]
        trainLabelL2 = [trainLabelL[j]]
        testInputL2 = [testInputL[j]]
        testLabelL2 = [testLabelL[j]]

        #The predicted test label of previous day becomes train imput of next day
        (trainInputL[j+1][1227],testInputL[j+1][1227]) = runDayByDay(trainInputL2, trainLabelL2, testInputL2,testLabelL2, shuffleTrain, batchSizeTrain, batchSizeTest,
                                                                          input_size,hidden_size, nLayer, output_size, dropout1,
                                                                          normalize, ss, n_epochs, lrRnnAdam, seqLengthInput,
                                                                          seqLengthLabel)

        outWinL.append(testInputL[j+1][1227]) #save the prediction of each test day

    # predict last day, which is 30th day
    trainInputL2 = [trainInputL[29]]
    trainLabelL2 = [trainLabelL[29]]
    testInputL2 = [testInputL[29]]
    testLabelL2 = [testLabelL[29]]

    (predicLastDayTrain,predicLastDayTest) = runDayByDay(trainInputL2, trainLabelL2, testInputL2,
                                                                      testLabelL2,
                                                                      shuffleTrain, batchSizeTrain, batchSizeTest,
                                                                      input_size,
                                                                      hidden_size, nLayer, output_size, dropout1,
                                                                      normalize, ss,
                                                                      n_epochs, lrRnnAdam, seqLengthInput,
                                                                      seqLengthLabel)
    predicLastDayTest2 = predicLastDayTest.reshape(predicLastDayTest.shape[0],)
    outWinL2 =  outWinL + [predicLastDayTest2] #combine 29day-prediction and 30th-day-prediction
    outWin3 = np.array(outWinL2)
    outWin4 = outWin3.reshape(-1, outWin3.shape[1])

    #save the predicted 30 day
    np.savetxt(
        './out/predictedTest_stock{}_mimicKalmanModified_{}winInput{}winLabel_{}epoch.csv'.format(
            stock, winInput, winLabel, n_epochs), outWin4, delimiter=",")

if __name__ == "__main__":
    # execute only if run as a script
    main()


