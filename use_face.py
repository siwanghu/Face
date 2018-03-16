# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:57:33 2018

@author: siwanghu
"""
import tensorflow as tf
import data_307
import sys
import cv2
import dlib

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],strides = [1, 2, 2, 1], padding = 'SAME')


x = tf.placeholder(tf.float32,[None,64,64,3])
y_ = tf.placeholder("float",[None,6])

x_img=tf.reshape(x, [-1, 64, 64, 3])

W_conv1 = weight_variable([3, 3, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([3, 3, 128, 256])
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_fc1 = weight_variable([4*4*256, 1024])
b_fc1 = bias_variable([1024])
h_pool4_flat = tf.reshape(h_pool4, [-1, 4*4*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 6])
b_fc2 = bias_variable([6])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

predict = tf.argmax(y_conv, 1)

saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('.'))  

def is_face(image):  
    res = sess.run(predict, feed_dict={x: image, keep_prob:1.0})  
    return data_307.libs[res[0]][0].split("/")[2]

detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)  
   
while True:  
    _, img = cam.read()  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff  
        if key == 27:
            sys.exit(0)
            
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1,x2:y2]
        face = cv2.resize(face, (64,64))
        print(is_face([face]))

        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
        cv2.imshow('image',img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)
  
sess.close() 





