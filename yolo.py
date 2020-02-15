import numpy as np
import tensorflow as tf
import cv2
from timeit import default_timer as timer
import time
import matplotlib.pyplot as plt

alpha = 0.1
threshold = 0.3
iou_threshold = 0.5
result_list = None
classes = []



def netWork(inputShape):

	x = tf.placeholder('float32',shape=inputShape)
	conv_1 = conv_layer(1,x,64,7,2)
	pool_2 = pooling_layer(2,conv_1,2,2)
	conv_3 = conv_layer(3,pool_2,192,3,1)
	pool_4 = pooling_layer(4,conv_3,2,2)
	conv_5 = conv_layer(5,pool_4,128,1,1)
	conv_6 = conv_layer(6,conv_5,256,3,1)
	conv_7 = conv_layer(7,conv_6,256,1,1)
	conv_8 = conv_layer(8,conv_7,512,3,1)
	pool_9 = pooling_layer(9,conv_8,2,2)
	conv_10 = conv_layer(10,pool_9,256,1,1)
	conv_11 = conv_layer(11,conv_10,512,3,1)
	conv_12 = conv_layer(12,conv_11,256,1,1)
	conv_13 = conv_layer(13,conv_12,512,3,1)
	conv_14 = conv_layer(14,conv_13,256,1,1)
	conv_15 = conv_layer(15,conv_14,512,3,1)
	conv_16 = conv_layer(16,conv_15,256,1,1)
	conv_17 = conv_layer(17,conv_16,512,3,1)
	conv_18 = conv_layer(18,conv_17,512,1,1)
	conv_19 = conv_layer(19,conv_18,1024,3,1)
	pool_20 = pooling_layer(20,conv_19,2,2)
	conv_21 = conv_layer(21,pool_20,512,1,1)
	conv_22 = conv_layer(22,conv_21,1024,3,1)
	conv_23 = conv_layer(23,conv_22,512,1,1)
	conv_24 = conv_layer(24,conv_23,1024,3,1)
	conv_25 = conv_layer(25,conv_24,1024,3,1)
	conv_26 = conv_layer(26,conv_25,1024,3,2)
	conv_27 = conv_layer(27,conv_26,1024,3,1)
	conv_28 = conv_layer(28,conv_27,1024,3,1)
	fc_29 = fc_layer(29,conv_28,512,flat=True,linear=False)
	fc_30 = fc_layer(30,fc_29,4096,flat=False,linear=False)
	fc_31 = fc_layer(31,fc_30, 1470, flat=False, linear=True)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess,weights_file)


def conv_layer(idx,inputs,filters,size,stride):
	channels = inputs.get_shape()[3]
	weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
	biases = tf.Variable(tf.constant(0.1, shape=[filters]))

	pad_size = size//2
	pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
	inputs_pad = tf.pad(inputs,pad_mat)

	conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')
	conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')
	print('Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d'
			%(idx,size,size,stride,filters,int(channels)))
	return tf.maximum(alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

def pooling_layer(idx,inputs,size,stride):
	print ('Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride))
	return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

def fc_layer(idx,inputs,hiddens,flat = False,linear = False):
	input_shape = inputs.get_shape().as_list()
	if flat:
		dim = input_shape[1]*input_shape[2]*input_shape[3]
		inputs_transposed = tf.transpose(inputs,(0,3,1,2))
		inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
	else:
		dim = input_shape[1]
		inputs_processed = inputs
		weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
		biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
		print ('Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' 
								%(idx,hiddens,int(dim),int(flat),1-	int(linear)))
	if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
	ip = tf.add(tf.matmul(inputs_processed,weight),biases)
	return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')



