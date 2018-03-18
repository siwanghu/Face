# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:01:54 2018

@author: siwanghu
"""
import random
import cv2 
import numpy

size=64
husiwang="./data/husiwang/"      #100000
liuhao="./data/liuhao/"          #010000
lumengjie="./data/lumengjie/"    #001000
pengsurong="./data/pengsurong/"  #000100
liqiang="./data/liqiang/"        #000010
xialing="./data/xialing/"        #000001      

libs={0:[husiwang,"100000"],1:[liuhao,"010000"],2:[lumengjie,"001000"],3:[pengsurong,"000100"],4:[liqiang,"000010"],5:[xialing,"000001"]}

def random_without_same(mins,maxs):
    return random.sample(range(mins,maxs),1)[0]

def one_hot(lable):
    list=[]
    for ch in lable:
        list.append(int(ch))
    return list

def next_batch(count):
    imgs=[]
    labs=[]
    for _ in range(count):
        try:
            people_id=random_without_same(0,6)
            img_id=random_without_same(10,20)
            img=cv2.imread(libs[people_id][0]+str(img_id)+".jpg")
            img = cv2.resize(img, (size, size))
            imgs.append(img)
            labs.append(one_hot(libs[people_id][1]))
        except:
            pass
    return numpy.array(imgs).astype('float32')/255.0,numpy.array(labs)

def next_batch_test(count):
    imgs=[]
    labs=[]
    for _ in range(count):
        try:
            people_id=random_without_same(0,6)
            img_id=random_without_same(0,10)
            img=cv2.imread(libs[people_id][0]+str(img_id)+".jpg")
            img = cv2.resize(img, (size, size))
            imgs.append(img)
            labs.append(one_hot(libs[people_id][1]))
        except:
            pass
    return numpy.array(imgs).astype('float32')/255.0,numpy.array(labs)
            
            
            
            