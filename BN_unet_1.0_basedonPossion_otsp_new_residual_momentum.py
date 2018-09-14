##################################################################################
### The Iterative Convolutional Neural Network (IterCNN) method is described in
###   Gong, Kuang, et al. "Iterative PET Image Reconstruction Using Convolutional 
###   Neural Network Representation." arXiv preprint arXiv:1710.03344 (2017).
##################################################################################
### Programmer: Kuang Gong @ MGH and UC DAVIS, 
### Contact: kgong@mgh.harvard.edu, kugong@ucdavis.edu
### Last Modified: 09-13-2018
### Note: This version is based on 3D U-net (detailed in our newly accepted TMI paper), 
### results shown in arXiv paper is based on 2D U-net. ---09-13-2018
##################################################################################

from __future__ import division, print_function, absolute_import
#import tensorflow as tf,tqdm
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

import numpy as np
#from PIL import Image
#import matplotlib.pyplot as plt
import os.path
import math
from collections import OrderedDict
#from pylab import *
import os
import sys


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def corrupt(x):
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))


def batch_relu(x, phase, scope):
    return tf.cond(phase,  
            lambda: tf.contrib.layers.batch_norm(x, is_training=True, decay=0.9, zero_debias_moving_mean=True,
                               center=False, updates_collections=None, scope=scope),  
            lambda: tf.contrib.layers.batch_norm(x, is_training=False,  decay=0.9, zero_debias_moving_mean=True,
                               updates_collections=None, center=False, scope=scope, reuse = True))  

#########################################################################
def weight_variable( shape, name):
    #initial = tf.truncated_normal(shape, stddev=stddev)
    n_input=shape[2]
    initial= tf.random_uniform(shape,-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input),name = name + 'initial')
    return tf.get_variable(name = name, initializer = initial)

def weight_variable_devonc( shape, name):
    #return tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    n_input=shape[2]
    initial= tf.random_uniform(shape,-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input), name = name + 'initial')
    return tf.get_variable(name = name, initializer = initial)

def bias_variable( shape, name):
    initial = tf.constant(0.00001, shape=shape)
    return tf.get_variable(name = name, initializer = initial)

def conv2d( x, W,keep_prob_, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)

def conv2d_stride( x, W,keep_prob_, name):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME', name = name)

def deconv2d( x, W,stride, name):
    x_shape = tf.shape(x,name = name + 'x_shape')
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2,  x_shape[3]//2], name = name + 'out_shape')
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID', name = name)


# In[2]:

#########################################################################
##### Parameters
#########################################################################
s1=128
s2=128
depth = 49
channels=1
num_p = 48
layers=4
filter_size=3
pool_size=2
features_root=16
keep_prob=1.0
n_class=1
n_epochs = 1000

learning_rate = 1e-3
display_step = 1
n_examples = 10
num_axisslice = depth
np.random.seed(0)
cost_try = np.zeros((n_epochs,1))



#########################################################################
##### Define graph
#########################################################################
# tf Graph input (only pictures)
X = tf.placeholder("float", [1, depth,s1,s2,channels],name='X')
Y = tf.placeholder("float", [1, depth,s1,s2, 1],name='Y')
phase = tf.placeholder(tf.bool,name='phase')
corruption=False

# Optionally apply denoising autoencoder
if corruption:
    current_input = corrupt(current_input)
# Build the encoder
weights = []
biases = []
convs = []
pools = OrderedDict()
deconv = OrderedDict()
dw_h_convs = OrderedDict()
up_h_convs = OrderedDict()
in_size = 3000
size = in_size
in_node=X
y_tensor=Y

for layer in range(0, layers):
    features = 2**layer*features_root    
    conv1 = tf.layers.conv3d(in_node, features, filter_size, padding='same', name='conv1_%d'%layer)
    batchn = batch_relu(conv1, phase,scope='bn%d_1'%(layer+1))
    dw_h_convs[layer]=lrelu(batchn,name= 'relu1_lay%d'%layer)
    conv2 = tf.layers.conv3d(dw_h_convs[layer], features, filter_size,strides=(1, 2, 2),
                             padding='same', name='conv2_%d'%layer)
    batchn = batch_relu(conv2, phase,scope='bn%d_2'%(layer+1))
    tmp_h_conv=lrelu(batchn,name= 'relu2_lay%d'%layer)    
    if layer < layers-1:
        in_node =tmp_h_conv
in_node = dw_h_convs[layers-1]
print(in_node)    
# up layers
for layer in range(layers-2, -1, -1):
    features = 2**(layer+1)*features_root
    in_node = tf.squeeze(in_node, [0])
    upsample1 = tf.image.resize_images(in_node, size=[in_node.get_shape().as_list()[1]*2,in_node.get_shape().as_list()[1]*2],
                                       method=tf.image.ResizeMethod.BILINEAR)
    upsample1 = tf.reshape(upsample1, [1, depth, in_node.get_shape().as_list()[1]*2, in_node.get_shape().as_list()[2]*2, features])
    upsample1 = tf.layers.conv3d(upsample1, features//2, filter_size, padding='same', name='down_conv1_%d'%layer)
    batchn = batch_relu(upsample1, phase,scope='up0_bn%d'%(layer+1))
    h_deconv = lrelu(batchn,name = 'down_relu1_lay%d'%layer)
    h_deconv_concat = tf.add(dw_h_convs[layer], h_deconv, name='add%d'%(layer+1))
    print("layer is %d"%layer)
    print(in_node)    
    deconv[layer] = h_deconv_concat
    conv1 = tf.layers.conv3d(h_deconv_concat, features//2, filter_size, padding='same', name='down_conv2_%d'%layer)
    batchn = batch_relu(conv1, phase,scope='up1_bn%d'%(layer+1))
    h_conv = lrelu(batchn,name = 'down_relu2_lay%d'%layer)
    conv2 = tf.layers.conv3d(h_conv, features//2, filter_size, padding='same', name='down_conv3_%d'%layer)
    batchn = batch_relu(conv2, phase,scope='up2_bn%d'%(layer+1))
    in_node = lrelu(batchn,name = 'down_relu3_lay%d'%layer)
    up_h_convs[layer] = in_node

# Output Map
output_map = tf.layers.conv3d(in_node, n_class, filter_size, padding='same', name='final_conv', activation = tf.nn.relu)




# In[3]:

# now have the reconstruction through the network
y_out = output_map
# cost function measures pixel-wise difference
cost = tf.reduce_sum(tf.square(y_out - Y))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# Create a saver for writing training checkpoints.
evaluate=tf.reduce_sum(tf.square(y_out - Y))
saver = tf.train.Saver(max_to_keep=30)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())


###########################################################
# We create a session to use the graph
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
saver.restore(sess,'../../pretraining_process/testLossNocomprz144Fea16Lay4it600')
print("Model restored.")
############# First define the graph for gradient computing
grads_wrt_input_tensor = tf.gradients(cost, X)
fp=open('image_evolve.img','rb')
test_xs=np.fromfile(fp,dtype=np.float32).reshape((1,num_axisslice,s1,s1,1))
zn = np.copy(test_xs);
#read the image 
test_ys=np.zeros((num_axisslice,s1,s1))
fp=open('image_input.img','rb')
test_ys=np.fromfile(fp,dtype=np.float32).reshape((1,num_axisslice,s1,s1,1))
test_ys = np.reshape(test_ys, (1,num_axisslice, s1, s1, 1))

learning_update = 0.001;

#learning_update = 0.01;
num_iter = int(sys.argv[2])

cost_momentum = np.zeros((num_iter,1))
tn = 1.0
for iter in range(num_iter):
    ############# Note, the input to the network should be zn when using momentum, not test_xs!!!
    ############# Below is to form into 9 channels from the input 
    tempzn = zn
    ############# Below is to compute the first order gradient using tf
    recon,input_grad = sess.run([y_out,grads_wrt_input_tensor], feed_dict={X: tempzn,Y:test_ys, phase:0}) 
    gradients = np.array(input_grad, dtype=np.float32)
    gradients = np.reshape(gradients, (1,num_axisslice, s1, s1, 1))    
    ############# Below is to update the input
    tnp1 = (1+np.sqrt(1+4 * tn * tn)) / 2.0
    test_xs_pre = np.copy(test_xs);
    test_xs = zn - learning_update * gradients
    test_xs[test_xs < 0]=0
    cost_momentum[iter] = np.sum(np.square(test_ys - recon))
    print("Iteration:", '%04d' % (iter+1),"cost=", "{:.9f}".format(float(cost_momentum[iter])))
    zn = test_xs + (tn -1.0) / tnp1 * (test_xs - test_xs_pre)  
    tn = np.copy(tnp1)
    
############# Below is to form into 9 channels from the input 
temptx = test_xs
recon = sess.run([y_out], feed_dict={X: temptx,Y:test_ys, phase:0}) 
output_img = np.array(recon, dtype = np.float32)
output_img = np.reshape(output_img, (num_axisslice,s1, s1))
fp=open('image_output.img','wb')
output_img.tofile(fp)
# to remove the old evolve image
cmd = 'rm image_evolve.img' 
os.system(cmd)
evolve_img = np.array(test_xs, dtype = np.float32)
evolve_img = np.reshape(evolve_img, (num_axisslice,s1, s1))
fp=open('image_evolve.img','wb')
evolve_img.tofile(fp)
myiter = sys.argv[1]
np.savetxt('cost_stride2_'+myiter+'.txt', cost_momentum, fmt='%1.5e', delimiter=',')







