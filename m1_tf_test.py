#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: m1_tf_test.py
# Created Date: Saturday July 9th 2022
# Author: Smiril
# Email: sonar@gmx.com
#############################################################
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Input
tf.__version__
tf.config.list_physical_devices()
from random import randrange

class ComputeSum(Layer):

  def __init__(self, input_dim):
      super(ComputeSum, self).__init__()
      # Create a non-trainable weight.
      self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                               trainable=True)

  def call(self, inputs):
      self.total.assign_add(tf.reduce_sum(inputs, axis=0))
      return self.total

  def call(self, data, depth = 0):
        n = len(data)
        return n

def ComputeSumModel(input_shape):
    inputs = Input(shape = input_shape)
    outputs = ComputeSum(input_shape[0])(inputs)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model

def xatoi(Str):
  
    sign, base, i = 1, 0, 0
      
    # If whitespaces then ignore.
    while (Str[i] == ' '):
        i += 1
      
    # Sign of number
    if (Str[i] == '-' or Str[i] == '+'):
        sign = 1 - 2 * (Str[i] == '-')
        i += 1
  
    # Checking for valid input
    while (i < len(Str) and 
          Str[i] >= '0' and Str[i] <= '9'):
                
        # Handling overflow test case
        if (base > (sys.maxsize // 10) or
           (base == (sys.maxsize // 10) and 
           (Str[i] - '0') > 7)):
            if (sign == 1):
                return sys.maxsize
            else:
                return -(sys.maxsize)
          
        base = 10 * base + (ord(Str[i]) - ord('0'))
        i += 1
      
    return base * sign

inputs = tf.keras.Input(shape=(xatoi(sys.argv[1]),), name="digits")
model = tf.keras.models.load_model('model')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / randrange(255+1), x_test / randrange(255+1)
model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(input_shape=(28, 28)),
 tf.keras.layers.Dense(128,activation='selu',name='layer1'),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(64,activation='relu',name='layer2'),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(32,activation='elu',name='layer3'),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(16,activation='tanh',name='layer4'),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(8,activation='sigmoid',name='layer5'),
 tf.keras.layers.Dropout(0.2)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
 loss=loss_fn,
 metrics=['accuracy'])
these = model.fit(x_train, y_train, epochs=10)
that = ComputeSum(len(these))
those = that(these)
outputs = tf.keras.layers.Dense(4, activation='softmax', name='predictions')(those)
model = ComputeSumModel((tf.shape(these)[1],))
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.build()
model.save('model')
model.summary()
