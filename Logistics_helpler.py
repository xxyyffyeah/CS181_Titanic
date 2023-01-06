import numpy as np


class LogisticsModel:
    def  __init__(self,_ground,_input):
        self.inputData=_input
        self.groundTruth=_ground.reshape((_ground.shape[0],1))
        self.inputNum=self.inputData.shape[0]
        self.w=np.ones((self.inputData.shape[1],1))
        self.learningRate=0.001
        self.itertimes=100000

    def Sigmoid(self,_z):
        return 1/(1+np.exp(-_z))
    def RunModel(self,_bacthdata):
        h=np.dot(_bacthdata,self.w)
        z=self.Sigmoid(h)
        return z
    def LossFunction(self,_ground,_predict):
        batchNum=_predict.shape[0]
        loss=-_ground*np.log(_predict)-(1-_ground)*np.log(1-_predict)
        loss=np.sum(loss)/batchNum
        return loss
    def GradientDecent(self,_ground,_batchdata):
        num=_batchdata.shape[0]
        # h =np.dot(_batchdata,self.w)
        z=self.RunModel(_batchdata)
        dw=(1.0/num)*np.dot(_batchdata.T,(z-_ground))
        self.w=self.w-dw*self.learningRate
    def Train(self):
        for i in range(self.itertimes):
            batchData=self.inputData
            predict=self.RunModel(batchData)
            self.GradientDecent(self.groundTruth,batchData)
            loss=self.LossFunction(self.groundTruth,predict)
            # print(loss)



