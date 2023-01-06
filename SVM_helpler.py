import random
import  numpy  as np
import matplotlib.pyplot as plt



class SVMModel:
    def  __init__(self,_ground,_input,_C,_toler,_maxIterTime,_delta):
        self.C=_C
        self.inputData=np.mat(_input)
        self.groundTruth=np.mat(_ground).transpose()
        self.toler=_toler
        self.b=0
        self.maxIterTime=_maxIterTime
        self.sampleNum,self.featureNum=np.shape(self.inputData)
        self.alphas = np.mat(np.zeros((self.sampleNum, 1)))
        self.K = np.mat(np.zeros((self.sampleNum, self.sampleNum)))
        for i in range(self.sampleNum):
            self.K[:, i] = self.KernelFunc(self.inputData[i, :], _delta)
    def PickIndex(self, _a, _b):
        ret = _a
        while ret == _a:
            ret = int(random.uniform(0, _b))
        return ret

    def ClipAlpha(self, _a, _H, _L):
        if _a > _H:
            _a = _H
        if _L > _a:
            _a = _L
        return _a

    def SMO(self):
        iter_num = 0
        while (iter_num < self.maxIterTime):
            alphaPairsChanged = 0
            for i in range(self.sampleNum):
                yi_predict =self.Predict_useAlpha(i)
                errori = yi_predict - float(self.groundTruth[i])
                if ((self.groundTruth[i] * errori < -self.toler) and (self.alphas[i] < self.C)) or (
                        (self.groundTruth[i] * errori > self.toler) and (self.alphas[i] > 0)):
                    j = self.PickIndex(i, self.sampleNum)
                    yj_predict = self.Predict_useAlpha(j)
                    errorj = yj_predict - float(self.groundTruth[j])
                    alphaiOld = self.alphas[i].copy();
                    alphajOld = self.alphas[j].copy();
                    if (self.groundTruth[i] != self.groundTruth[j]):
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0,self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])
                    if L == H:
                        continue
                    # eta = 2.0 * self.inputData[i, :] * self.inputData[j, :].T \
                    #       - self.inputData[i, :] * self.inputData[i, :].T \
                    #       - self.inputData[j,:] * self.inputData[j,:].T
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] -= self.groundTruth[j] * (errori - errorj) / eta

                    self.alphas[j] =self.ClipAlpha(self.alphas[j], H, L)

                    if (abs(self.alphas[j] - alphajOld) < 0.0001):
                        continue
                    self.alphas[i] += self.groundTruth[j] * self.groundTruth[i] * (alphajOld - self.alphas[j])

                    # b1 = self.b - errori - self.groundTruth[i] * (self.alphas[i] - alphaiOld) * self.inputData[i, :] * self.inputData[i, :].T - \
                    #      self.groundTruth[j] * (self.alphas[j] - alphajOld) * self.inputData[i, :] * self.inputData[j, :].T
                    # b2 = self.b - errorj - self.groundTruth[i] * (self.alphas[i] - alphaiOld) * self.inputData[i, :] * self.inputData[j, :].T - \
                    #      self.groundTruth[j] * (self.alphas[j] - alphajOld) * self.inputData[j, :] * self.inputData[j, :].T
                    b1 = self.b - errori - self.groundTruth[i] * (self.alphas[i] - alphaiOld) * self.K[i,i] - \
                         self.groundTruth[j] * (self.alphas[j] - alphajOld) * self.K[i,j]
                    b2 = self.b - errorj - self.groundTruth[i] * (self.alphas[i] - alphaiOld) * self.K[i,j]- \
                         self.groundTruth[j] * (self.alphas[j] - alphajOld) * self.K[j,j]

                    if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                        self.b = b1
                    elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    alphaPairsChanged += 1
            if (alphaPairsChanged == 0):
                iter_num += 1
            else:
                iter_num = 0
        self.GetW()
    def KernelFunc(self, A,p):
        K = np.mat(np.zeros((self.sampleNum,1)))
        for i in range(self.sampleNum):
            tempR = self.inputData[i,:] - A
            K[i] = tempR * tempR.T
        K = np.exp(K / (-1 * p ** 2))
        return K

    def GetW(self):
        alphas, dataMat, groudTruth = np.array(self.alphas), np.array(self.inputData), np.array(self.groundTruth.transpose())
        self.w = np.dot((np.tile(groudTruth.reshape(1, -1).T, (1,self.featureNum)) * dataMat).T, alphas)

    def Predict_useAlpha(self,_index):
        # return float(np.multiply(self.alphas, self.groundTruth).T * (self.inputData * self.inputData[_index, :].T)) +self.b
        return float(np.multiply(self.alphas, self.groundTruth).T * self.K[:,_index]) + self.b