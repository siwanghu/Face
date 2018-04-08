# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:01:54 2018

@author: siwanghu
"""
import tensorflow as tf
#import data_307
import att_orl

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
y_ = tf.placeholder("float",[None,40])

x_img=tf.reshape(x, [-1, 64, 64, 3])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([8*8*128, 1024])
b_fc1 = bias_variable([1024])
h_pool4_flat = tf.reshape(h_pool3, [-1, 8*8*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 40])
b_fc2 = bias_variable([40])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))

tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

step,steps=0,2000

while step<steps:
    images,labs = att_orl.next_batch(50)
    _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],feed_dict={x:images,y_:labs, keep_prob:0.5})
    if step%10==0:
        acc = accuracy.eval({x:att_orl.x_test, y_:att_orl.y_test, keep_prob:1.0})
        print("the %d step:,acc:" % step ,acc)
    step=step+1
    if acc >= 0.95:
        saver.save(sess, './train_faces.model', global_step=step*20)
        step=steps+500
if step==steps:
    saver.save(sess, './train_faces.model', global_step=step*20)