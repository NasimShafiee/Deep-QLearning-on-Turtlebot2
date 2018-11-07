#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym import wrappers
import gym_gazebo
import time
import numpy as np
import random
import time
import cv2
import tensorflow as tf
from skimage import transform 

import qlearn
import liveplot

tf.logging.set_verbosity(tf.logging.INFO)

#to create our NN function
def cnn_model_fn(features, labels, mode): #features:inputs/ labels:outputs/ modes:Precition OR evaluation
  """Model function for CNN."""
  ## Input Layer ---------------------------------------------------------------------------------------------------
  input_layer = tf.reshape(features["x"], [-1, 32,32, 1]) 
  ## Convolutional Layer #1 ----------------------------------------------------------------------------------------
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  #output:[batch_size, 32,32, 32]
  
  ## Pooling Layer #1 ----------------------------------------------------------------------------------------------
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  #output:[batch_size, 16,16, 32]
  
  ## Convolutional Layer #2 ----------------------------------------------------------------------------------------
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  #output:[batch_size, 16,16, 64]

  ## Pooling Layer #1 ----------------------------------------------------------------------------------------------
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  #output tensor of pooling2d() : shape of [batch_size, 8,8, 64]

  ## Dense Layer ---------------------------------------------------------------------------------------------------
  pool2_flat = tf.reshape(pool2, [-1, 8*8*64])
  #output:[batch_size, 8*8*64]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  #output:[batch_size, 1024]
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  #output:[batch_size, 1024]

  ## Logits Layer --------------------------------------------------------------------------------------------------
  logits = tf.layers.dense(inputs=dropout, units=3)

  
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) #stochastic gradient descent
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#####################################################################################################################



def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':

    env = gym.make('GazeboCircuit2TurtlebotLidar-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)

    last_time_steps = np.ndarray(0)

    epsilon_discount = 0.9986

    start_time = time.time()
    total_episodes = 10000
    highest_reward = 0
    cv2.namedWindow("reduced_image", 1)
    #reset the training graph
    tf.reset_default_graph()

    #feed-forward part
    inputs1 = tf.placeholder(shape=[None,1024],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([1024,3],0,0.01))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)

    #loss evaluation
    nextQ = tf.placeholder(shape=[1,3],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)
    #initialize network
    init = tf.global_variables_initializer()

    ## Create the Estimator/Classifier ----------------------------------------------------------------------------------
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/home/nasim/Desktop/TurtlebotTraining")
    ## Set up logging for predictions -----------------------------------------------------------------------------------
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    tensors_to_log1 = {"probabilities": "predicted_tensor"}
    logging_hook1 = tf.train.LoggingTensorHook(tensors=tensors_to_log1, every_n_iter=50)
    print("SETTING UP DONE")
###############################
###############################
###############################
    #Q_func=tf.placeholder(shape=(1,3),dtype=np.float32)
    Q_func=np.zeros((1,3))
    action=tf.placeholder(shape=(1),dtype=tf.int32)
    with tf.Session() as sess:
    	sess.run(init)
    	for x in range(total_episodes):
        	done = False
        	cumulated_reward = 0 
        	observation,info = env.reset()
        	image1 = cv2.resize(info, (32,32), interpolation = cv2.INTER_NEAREST)                       		
        	image1 = tf.image.convert_image_dtype(image1, dtype=tf.float32)
        	image1  = tf.reshape(image1, shape=(1,1024))
        	print(np.shape(image1))
        	#if qlearn.epsilon > 0.05:
            	#	qlearn.epsilon *= epsilon_discount
        	
        	#render() #defined above, not env.render()
        	state = ''.join(map(str, observation))
        	#print("state:-------------",state);
        	for i in range(1):#1500
			action=mnist_classifier.predict(
            		  		input_fn=image1,
            		  		hooks=[logging_hook1])

            		act=np.zeros(1) #tf.argmax(Q_func,1)

			print(action,"--------------------ACTION SELECTED------------------------")
            		observation, reward, done, info = env.step(action) 
			print("--------------------ACTION DONE------------------------")
            		image1 = cv2.resize(info, (32,32), interpolation = cv2.INTER_NEAREST) 
            		image1 = tf.image.convert_image_dtype(image1, dtype=tf.float32)  
            		image1  = tf.reshape(image1, shape=(1,1024))
			print("--------------------IMAGE CONVERTED------------------------")
            		train_input_fn = tf.estimator.inputs.numpy_input_fn(
             		  x={"x": image1.eval()},
            		  y=act, 
            		  batch_size=1,
            		  num_epochs=None,
            		  shuffle=True)
			print("--------------------INPUT READY------------------------")
            		mnist_classifier.train(
            		  input_fn=train_input_fn,
            		  steps=3,
            		  hooks=[logging_hook])


			print("--------------------ACTION TRAINED------------------------")
            		cumulated_reward += reward
            		if highest_reward < cumulated_reward:
                		highest_reward = cumulated_reward
			
            		nextState = ''.join(map(str, observation))
            		
            		env._flush(force=True)
            		if not(done):
                		state = nextState
            		else:
                		last_time_steps = np.append(last_time_steps, [int(i + 1)])
                		break

        	if x%10==0:
            		plotter.plot(env)

        	m, s = divmod(int(time.time() - start_time), 60)
        	h, m = divmod(m, 60)
        	print ("EP: "+str(x+1)+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

   	#Github table content
    	#print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str		(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

    	l = last_time_steps.tolist()
    	l.sort()

    #print("Parameters: a="+str)
    	print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    	print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    	env.close()
