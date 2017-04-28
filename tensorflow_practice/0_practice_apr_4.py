#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np, findspark
findspark.init()
import pyspark

# 0. It seems like we make the placeholder first then do the computation
a = tf.placeholder("float") # Create a symbolic variable 'a'
b = tf.placeholder("float") # Create a symbolic variable 'b'

# 1. this is the multiplicaiton
y = tf.multiply(a, b) # multiply the symbolic variables

# Document of operations possible  - https://www.tensorflow.org/api_guides/python/math_ops
yy = tf.div(a, b)

with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
    print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))
    print("%f should equal 1.0 - division" % sess.run(yy, feed_dict={a: 3, b: 3}))

print "\n\n\n\n"
# following the placeholder declaration
image_size = 32
image_dim = image_size * image_size * 3
xs = tf.placeholder("float", shape=[None, image_dim])
print xs
# And it seems like we do not need to put a dimension in the front - None

# Place holder example from tensorflow website
x = tf.placeholder(tf.float32, shape=(3, 3))
y = tf.matmul(x, x)
with tf.Session() as sess:
  # print(sess.run(y))  # ERROR: will fail because x was not fed.
  rand_array = np.random.rand(3, 3)
  print rand_array,"\n\nCreated Matrix\n\n"
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.


# Place holder example from tensorflow website - the NONE makes the matrix dynmaic
x = tf.placeholder(tf.float32, shape=(None, 3))
x2 = tf.placeholder(tf.float32, shape=(None, 3))
y = tf.matmul(x, x2)
with tf.Session() as sess:
  # print(sess.run(y))  # ERROR: will fail because x was not fed.
  rand_array = np.random.rand(5, 3)
  rand_array_2 = np.random.rand(3, 3)
  print rand_array,"\n\nCreated Matrix - changed Shape \n\n"
  print(sess.run(y, feed_dict={x: rand_array ,x2:  rand_array_2 }  ))  # Will succeed.

# ----- END OF THE CODE -------