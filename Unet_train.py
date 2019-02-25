
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
import matplotlib.pyplot as plt
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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#from IPython import display
#get_ipython().magic(u'matplotlib inline')
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


# In[ ]:

#########################################################################
##### Parameters
#########################################################################
s1=144
s2=144
depth = 96
channels=1
layers=4
filter_size=3
pool_size=2
features_root=16
keep_prob=1.0
n_class=1
num_p = 42

learning_rate = 1e-3
batch_size =1
n_epochs =2000
display_step = 1
n_examples = 10
num_axisslice = depth
np.random.seed(0)
cost_try = np.zeros((n_epochs*num_p,1))
cost_test = np.zeros((n_epochs*num_p,1))
cost_total = np.zeros((n_epochs*num_p,1))
sample_total = np.zeros((n_epochs*num_p,1))



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


# In[ ]:

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

###########################################################################
### Print out the number of trainable parameters
###########################################################################
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)
###########################################################################
### Below is to load the data for training
###########################################################################
fp=open('leave1710.gesigna.input.ge.img.144x144x96.45.p10test.permute.iter357','rb')
Train= np.fromfile(fp,dtype=np.float32).reshape((num_p, depth,s1,s2,channels))
Train = np.reshape(Train, (num_p, depth,s1,s2,channels))

fp=open('leave1710.gesigna.label.ge.img.144x144x96.45.p10test.permute.iter357','rb')
Truth=np.fromfile(fp,dtype=np.float32).reshape((num_p, depth,s1,s2,1))
Truth = np.reshape(Truth, (num_p, depth,s1,s2,1))
###########################################################################
### Below is to load the data for validataion
###########################################################################
fp=open('vali.leave1710.gesigna.input.ge.img.144x144x96.45.p10test.permute.iter357','rb')
Train_vali= np.fromfile(fp,dtype=np.float32).reshape((3, depth,s1,s2,channels))
Train_vali = np.reshape(Train_vali, (3, depth,s1,s2,channels))

fp=open('vali.leave1710.gesigna.label.ge.img.144x144x96.45.p10test.permute.iter357','rb')
Truth_vali=np.fromfile(fp,dtype=np.float32).reshape((3, depth,s1,s2,1))
Truth_vali = np.reshape(Truth_vali, (3, depth,s1,s2,1))


###########################################################################
### Below is to train the network
###########################################################################
# Fit all training data
batch_size = 1
total_batch = int(num_p/batch_size)
for epoch_i in range(n_epochs):
    # Loop over all batches
    sample_all=np.random.randint(0,Train.shape[0],Train.shape[0])
    cost_temp=0
    for batch_i in range(total_batch):
        bstart = batch_i * batch_size
        bend = (batch_i + 1)* batch_size
        sample=sample_all[bstart : bend]
        batch_xs= Train[sample]
        batch_ys= Truth[sample]
        _, c = sess.run([train_step, cost], feed_dict={X: batch_xs, Y: batch_ys,  phase:1})
        cost_try[epoch_i*num_p + batch_i] = c
        sample_total[epoch_i*num_p + batch_i] = sample
        print("Epoch:", '%04d' % (epoch_i+1),"Sample:", '%04d' % (sample+1),"cost=", "{:.9f}".format(c))        
        batch_xs= np.reshape(Train_vali[0], (1,depth,s1,s2,1))
        batch_ys= np.reshape(Truth_vali[0], (1,depth,s1,s2,1))
        evl1 = sess.run(evaluate, feed_dict={X: batch_xs, Y: batch_ys, phase:0})
        batch_xs= np.reshape(Train_vali[1], (1,depth,s1,s2,1))
        batch_ys= np.reshape(Truth_vali[1], (1,depth,s1,s2,1))
        evl2 = sess.run(evaluate, feed_dict={X: batch_xs, Y: batch_ys, phase:0})        
        batch_xs= np.reshape(Train_vali[2], (1,depth,s1,s2,1))
        batch_ys= np.reshape(Truth_vali[2], (1,depth,s1,s2,1))
        evl3 = sess.run(evaluate, feed_dict={X: batch_xs, Y: batch_ys, phase:0})
        test_all = evl1 + evl2 + evl3
        cost_test[epoch_i*num_p + batch_i]=test_all     
        print("Test=", "{:.9f}".format(test_all))        
    if (epoch_i +1) % 50 == 0:
        saver.save(sess, './testLossNocomprz144Fea'+str(features_root)+'Lay'+str(layers)+'it' + str(epoch_i +1))
        np.savetxt('./Layer4tryLossNocomprz144Fea'+str(features_root)+'Lay'+str(layers)+'it'+str(epoch_i +1)+'.txt', cost_try, fmt='%d', delimiter=',')
        np.savetxt('./Layer4AlltestLossCostNocomprz144Fea'+str(features_root)+'Lay'+str(layers)+'it'+str(epoch_i +1)+'.txt', cost_test, fmt='%d', delimiter=',')
        np.savetxt('./Layer4AllSampleNocomprz144Fea'+str(features_root)+'Lay'+str(layers)+'it'+str(epoch_i +1)+'.txt', sample_total, fmt='%d', delimiter=',')

print("Optimization Finished!")








