# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:13:42 2018

@author: siwanghu
"""
import cv2
import glob
import random
import numpy as np

att_faces=".\\backup"    #图片保存路径
trainNum=5               #每个人训练集图片张数
feature=50               #PCA特征分解取值个数

def loadATTData():
    x_train,y_train,x_test,y_test=[],[],[],[]
    for i in range(40):
        path=att_faces+"\\s"+str(i+1)+"\\*.pgm"
        imgs=[cv2.imread(img) for img in glob.glob(path)]
        randoms=random.sample(range(10),trainNum)
        x_train.extend([imgs[num].ravel() for num in range(10) if num in randoms])
        x_test.extend([imgs[num].ravel() for num in range(10) if num not in randoms])
        y_train.extend([i]*trainNum)
        y_test.extend([i]*(10-trainNum))
    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

def PCA(data):
    X=np.float32(np.mat(data))
    U=np.mean(X,0)
    Z=X-np.tile(U,(X.shape[0],1))
    D,V=np.linalg.eig(Z*Z.T)
    vector=V[:,:feature]
    vector=Z.T*vector
    for i in range(feature):
        vector[:,i] /= np.linalg.norm(vector[:,i])  
    return np.array(Z*vector),U,vector 

x_train,y_train,x_test,y_test=loadATTData()
xtrain,mean,vector = PCA(x_train)  

xTest = np.array((x_test-np.tile(mean,(x_test.shape[0],1))) * vector)
yPredict =[y_test[np.sum((xtrain-np.tile(d,(x_train.shape[0],1)))**2, 1).argmin()] for d in xTest]  
print("识别正确率: %.2f%%"% ((yPredict == np.array(y_test)).mean()*100))