import tensorflow as tf 
import numpy as np
import matplotlib

# 0. Using the numpy x
x = np.linspace(-3,3,100)
print x.shape,x.dtype

# 1. Using the tf 
xx = tf.linspace(-3.0,3.0,100,name="Wow_can_even_give_names....")
print xx

# Get the default graph
g = tf.get_default_graph()

# Get the tensor object - actual object
graph = g.get_tensor_by_name("Wow_can_even_give_names....:0")
print "The tensor object : ", graph

# Get the session
sess = tf.Session()
print sess




# ---- END OF THE CODE -----