#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: m1_test.py
# Created Date: Saturday July 9th 2022
# Author: Smiril
# Email: sonar@gmx.com
#############################################################
import tensorflow as tf
tf.__version__
tf.config.list_physical_devices()
logits = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]
model = tf.keras.models.load_model('model')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(input_shape=(28, 28)),
 tf.keras.layers.Dense(128,activation='selu'),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(10,activation='tanh'),
 tf.keras.layers.Dropout(0.2),
 tf.keras.layers.Dense(2,activation='sigmoid'),
 tf.keras.layers.Dropout(0.2)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
 loss=loss_fn,
 metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
model.build()
model.save('model')
model.summary()
