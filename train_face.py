# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:01:54 2018

@author: siwanghu
"""
import tensorflow as tf
import data_307
import sys

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

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))

tf.summary.scalar('loss', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

imgs_test,labs_test=data_307.next_batch_test(50)
for i in range(3000):
    images,labs = data_307.next_batch(50)
    _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],feed_dict={x:images,y_:labs, keep_prob:0.5})
    acc = accuracy.eval({x:imgs_test, y_:labs_test, keep_prob:1.0})
    print("the %d step:,acc:" % i ,acc)

    if acc >= 0.95:
        saver.save(sess, './train_faces.model', global_step=i*50)
        sys.exit(0)

saver.save(sess, './train_faces.model', global_step=2000*50)

