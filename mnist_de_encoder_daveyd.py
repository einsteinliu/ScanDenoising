from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import framework
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope

tf.logging.set_verbosity(tf.logging.INFO)

def lenet_with_scope(features,labels,mode):
    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    groundtruth = tf.reshape(labels,[-1,28,28,1])
    sp = input_layer.get_shape()
    arg_scope_conv2d = tf.contrib.framework.arg_scope([layers.conv2d],
                                        kernel_size=[5,5],
                                        activation_fn = tf.nn.leaky_relu,
					                    normalizer_fn=layers.batch_norm)
    arg_scope_deconv2d = tf.contrib.framework.arg_scope([layers.conv2d_transpose],
                                        kernel_size=[2,2],
                                        stride = 2,
                                        activation_fn = tf.nn.leaky_relu,
					normalizer_fn=layers.batch_norm)

    arg_scope_full_connected = tf.contrib.framework.arg_scope([layers.fully_connected],
                                                              activation_fn=tf.nn.leaky_relu) 
    with arg_scope_conv2d:
        with arg_scope_full_connected:
            with arg_scope_deconv2d:
                conv1 = tf.contrib.layers.conv2d(input_layer,num_outputs=32) 
                pool1 = tf.contrib.layers.max_pool2d(conv1,kernel_size=[2,2],stride=2)
                conv2 = tf.contrib.layers.conv2d(pool1,num_outputs=64)
                pool2 = tf.contrib.layers.max_pool2d(conv2,kernel_size=[2,2],stride=2)
                pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

                dense1 = tf.contrib.layers.fully_connected(inputs=pool2_flat, num_outputs=2)
                dense2 = tf.contrib.layers.fully_connected(inputs=dense1, num_outputs=7*7*64)

                dense_unflat = tf.reshape(dense2,[-1,7,7,64])

                conv21 = tf.contrib.layers.conv2d(dense_unflat,num_outputs=64)
                deconv21 = tf.contrib.layers.conv2d_transpose(conv21,num_outputs=32)
                conv22 = tf.contrib.layers.conv2d(deconv21,num_outputs=32)
                reconstruct_logits = tf.contrib.layers.conv2d_transpose(conv22,num_outputs=1,
									     activation_fn=None,
									     normalizer_fn=None)                
                
                reconstruct_probs = tf.nn.sigmoid(reconstruct_logits,name = "softmax")

                tf.summary.image("Diff",tf.abs(reconstruct_probs-groundtruth))      
                tf.summary.image("Result",reconstruct_probs)
                tf.summary.image("Origin",groundtruth)

                reconstruction = {
                        "probabilities":reconstruct_probs
                        }

                if mode == tf.estimator.ModeKeys.PREDICT:
                    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
                
                loss = tf.losses.absolute_difference(labels=groundtruth,predictions=reconstruct_probs)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    optimizer = tf.train.AdamOptimizer(1e-4)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    training_op = optimizer.minimize(
                                loss = loss,
                                global_step = tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training_op)

def main(args):
    nums = []
    for i in range(10):
        img = np.zeros([28,28])        
        cv2.putText(img,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,1,1,2,cv2.LINE_AA)
        nums.append(np.float32(img.flatten()))

    mnist = input_data.read_data_sets('MNIST_data')
    train_image = mnist.train.images    
    train_labels = np.array(mnist.train.labels,dtype=np.int32)

    ##extract 10 pattern image
    #pattern_image = []
    #for i in range(0,10):
    #    j = 0
    #    while(train_labels[j]!=i):
    #        j = j + 1
    #    pattern_image.append(train_image[j])    
    
    train_image_labels = []
    #construct ground truth
    for i in range(0,train_labels.shape[0]):
        train_image_labels.append(nums[train_labels[i]])

    train_image_labels = np.asarray(train_image_labels)    
    #shp = np.shape(train_image_labels)

    estimator = tf.estimator.Estimator(
        model_fn = lenet_with_scope,        
        model_dir = "D:/Study/DeepLearning/MNIST_Playground/mnist-playground/ckp")

    tensors_to_log = {} #{"probabilities": "softmax"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
    
    train_func = tf.estimator.inputs.numpy_input_fn(
        x = {"x":train_image},
        y = train_image_labels,
        batch_size = 50,
        num_epochs=None,
        shuffle=True)

    estimator.train(
        input_fn = train_func,
        steps = 200000,
        hooks = [logging_hook])

if __name__ == "__main__":
  tf.app.run()
