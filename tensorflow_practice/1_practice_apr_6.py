import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

# 0. The range from -1 to 1 and there are 10 numbers between
trX = np.linspace(-1, 1, 10)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 
# create a y value which is approximately linear but with some random noise

# 1. Declare the variables
image_size = 32
image_dim = image_size * image_size * 3
label_count = 10

# 2. make the tensor flow place holder
xs = tf.placeholder("float", shape=[None, image_dim])
ys = tf.placeholder("float", shape=[None, label_count]) # Place holder for the dim that have a None, 10
lr = tf.placeholder("float", shape=[]) # This does not have a dim.....
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder("bool", shape=[])

print xs

# 3. The first operation in the xs
current = tf.reshape(xs, [ -1, 32, 32, 3 ])
print current
# current = conv2d(current, 3, 16, 3)

# Func - So this is what the weight_variable is..
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

# 4. what does the weight_variable do...?
W = weight_variable([ 3, 3, 3, 16 ])
print W

# 5. Cannot even understand that this truncated_normal is doing
temp = tf.truncated_normal([1,1],name="donin")
for x in dir(temp):
	t = getattr(temp,x)
	print x," : ", t








import sys
sys.exit(0)

# Func - of Performing 2D convolution
def conv2d(input, in_features, out_features, kernel_size, with_bias=False):
  W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
  conv = tf.nn.conv2d(input, W, [ 1, 1, 1, 1 ], padding='SAME')
  if with_bias:
    return conv + bias_variable([ out_features ])
  return conv

# ------- PRACTICE ZONE --------
a = tf.placeholder("float",shape= []) # Create a symbolic variable 'a'
b = tf.placeholder("float") # Create a symbolic variable 'b'

y = tf.multiply(a, b) # multiply the symbolic variables
with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    print("%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
    print("%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))

# ------ END OF THE CODE -------