# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:32:53 2018

@author: siwanghu
"""
import cv2
import os
import glob
import random
import numpy

classs=40                                       #图像类别
trainNum=8                                      #每个人训练集图片张数
att_faces=".\\att_faces"                        #图片保存路径
backup=".\\backup"                              #处理后的图像
randoms=random.sample(range(1,11),trainNum)       #随机取每个人的trainNum张图片作为训练集

def one_hot(labels):
    hot=[i-i for i in range(classs)]
    hot[labels-1]=1
    return hot

def resizeImg():
     for i in range(classs):
        path=att_faces+"\\s"+str(i+1)+"\\*.pgm"
        paths=glob.glob(path)
        for path in paths:
            img=cv2.imread(path)
            img=cv2.resize(img,(64,64))
            path=path.replace("att_faces","backup")
            if not os.path.exists(".\\backup\\s"+str(i+1)):
                os.makedirs(".\\backup\\s"+str(i+1))
            cv2.imwrite(path,img)

def next_batch(count):
    imgs=[]
    labs=[]
    peoples=[i+1 for i in range(classs)]
    for _ in range(count):
        try:
            people_id=random.choice(peoples)
            img_id=random.choice(randoms)
            img=cv2.imread(backup+"\\s"+str(people_id)+"\\"+str(img_id)+".pgm")
            imgs.append(img)
            labs.append(one_hot(people_id))
        except:
            print("read trainSet error!")
    return numpy.array(imgs).astype('float32')/255.0,numpy.array(labs)    

def next_batch_test():
    imgs=[]
    labs=[]
    for people_id in [i+1 for i in range(classs)]:
        for img_id in [i+1 for i in range(10)]:
            if img_id not in randoms:
                img=cv2.imread(backup+"\\s"+str(people_id)+"\\"+str(img_id)+".pgm")
                imgs.append(img)
                labs.append(one_hot(people_id))
    return numpy.array(imgs).astype('float32')/255.0,numpy.array(labs)

x_test,y_test=next_batch_test()