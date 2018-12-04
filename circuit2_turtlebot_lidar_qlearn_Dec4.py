#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import gym_gazebo
import tensorflow as tf
import numpy as np
import time
import random
from random import *
import cv2
from gym import wrappers
from skimage import transform 

import qlearn
import DQNTF
import liveplot

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def render():
	render_skip       = 0 #Skip first X episodes.
	render_interval   = 50 #Show render Every Y episodes.
	render_episodes   = 10 #Show Z episodes every rendering.

	if (x%render_interval == 0) and (x != 0) and (x > render_skip):
		env.render()
	elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
		env.render(close=True)


def cnn_model_fn(features, labels, mode):  # features:inputs/ Q:outputs/ modes:Prediction OR train
	"""Model function for CNN."""
	## Input Layer ---------------------------------------------------------------------------------------------------
	input_layer = tf.reshape(features["x"], [-1, 32, 32, 4])
	q_tmp       = tf.reshape(features["q"], [3, 1])
	q           = tf.transpose(q_tmp)
	## Convolutional Layer #1 ----------------------------------------------------------------------------------------
	conv1       = tf.layers.conv2d(
		inputs      = input_layer,
		data_format='channels_last',
		filters     = 32,#32
		kernel_size = [5, 5],
		padding     = "same",
		activation  = tf.nn.relu)
	# output:[batch_size, 32,32, 32]
	#print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&cov1 size",conv1.shape)
	## Pooling Layer #1 ----------------------------------------------------------------------------------------------
	pool1       = tf.layers.max_pooling2d(inputs=conv1, data_format='channels_last', pool_size=[2, 2], strides=2)
	# output:[batch_size, 16,16, 32]
	#print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&pool1 size", pool1.shape)
	## Convolutional Layer #2 ----------------------------------------------------------------------------------------
	conv2       = tf.layers.conv2d(
		inputs      = pool1,
		data_format='channels_last',
		filters     = 64,#64
		kernel_size = [5, 5],
		padding     = "same",
		activation  = tf.nn.relu)
	# output:[batch_size, 16,16, 64]
	#print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&cov2 size", conv2.shape)
	## Pooling Layer #1 ----------------------------------------------------------------------------------------------
	pool2       = tf.layers.max_pooling2d(inputs=conv2, data_format='channels_last', pool_size=[2, 2], strides=2)
	# output tensor of pooling2d() : shape of [batch_size, 8,8, 64]
	#print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&pool2 size", pool2.shape)
	## Dense Layer ---------------------------------------------------------------------------------------------------
	pool2_flat  = tf.reshape(pool2, [-1, 8 * 8 * 64])#*64
	# output:[batch_size, 8*8*64]
	dense       = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)#1024
	# output:[batch_size, 1024]
	dropout     = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	# output:[batch_size, 1024]
	
	## Logits Layer --------------------------------------------------------------------------------------------------
	logits = tf.layers.dense(inputs=dropout, units=3)
	# output:[1,3]
	
	predictions = {
		"classes":       tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
		"logits":        logits
	}
	#print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&testtttt", conv1.shape)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	# Calculate Loss (for both TRAIN and EVAL modes)
	# loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	loss = tf.losses.mean_squared_error(q, logits)
	
	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer   = tf.train.GradientDescentOptimizer(learning_rate=0.001)                            # stochastic gradient descent
		train_op    = optimizer.minimize(
			loss        = loss,
			global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
	##################################SET UP ENVIRONMENT#######################################
	env               = gym.make('GazeboCircuit2TurtlebotLidar-v0')
	outdir            = '/tmp/gazebo_gym_experiments'
	env               = gym.wrappers.Monitor(env, outdir, force=True)
	plotter           = liveplot.LivePlot(outdir)
	last_time_steps   = np.ndarray(0)
	qlearn            = qlearn.QLearn(actions=range(env.action_space.n),
	                                  alpha=0.2, gamma=0.8, epsilon=0.9)
	initial_epsilon   = qlearn.epsilon
	epsilon_discount  = 0.9986
	start_time        = time.time()
	total_episodes    = 10000
	highest_reward    = 0
	gamma             = 0.8
	num_actions       = 3
	action_space      = [0,1,2]
	
	#tf.logging.set_verbosity(tf.logging.INFO)
	tf.logging.set_verbosity(tf.logging.ERROR)
	tf.reset_default_graph()                                                                              # Reset training graph
	myinit            = tf.global_variables_initializer()                                                 # Initialize training network
	turtle_regressor  = tf.estimator.Estimator(model_fn=cnn_model_fn,
	                                           model_dir="/home/nasim/Desktop/TurtlebotDQN")              # Create the Estimator/Classifier
	
	tensors_to_log    = {"probabilities": "softmax_tensor"}                                               # Set up logging for predictions
	logging_hook      = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
	tensors_to_log1   = {"probabilities": "predicted_tensor"}
	logging_hook1     = tf.train.LoggingTensorHook(tensors=tensors_to_log1, every_n_iter=50)
	
	Q                 = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
	Q_nxt             = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
	action            = tf.placeholder(shape=(1),dtype=tf.int32)
	Q_s_a             = tf.placeholder(shape=(1),dtype=tf.int32)
	Q_s_a_n           = tf.placeholder(shape=(1),dtype=tf.int32)
	
	
	exploration_factor= 1.0
	with tf.Session() as sess:
		sess.run(myinit)
		
		for x in range(total_episodes):
			
			done              = False
			cumulated_reward  = 0 #Should going forward give more reward then L/R ?
			if qlearn.epsilon > 0.05:
				qlearn.epsilon    *= epsilon_discount
			#render() #defined above, not env.render()
			
			_ , info    = env.reset()
			img_tmp     = cv2.resize(info, (32, 32), interpolation=cv2.INTER_CUBIC)#INTER_NEAREST)#(32,32)
			#print("resize shape",img_tmp.shape)
			img_tmp     = tf.image.convert_image_dtype(img_tmp, dtype=tf.float32)
			state       = tf.reshape(img_tmp, shape=(1, 1024*4))
			state2      = tf.reshape(img_tmp, shape=(1, 1024*4))
			
			predict_input_fn = tf.estimator.inputs.numpy_input_fn(
					x={"x": np.array(state.eval()), "q": np.array([[0, 0, 0]])},                  # q does not affect to NN
					batch_size=1,
					num_epochs=1,
					shuffle=False)
			pred = list(turtle_regressor.predict(input_fn=predict_input_fn))
			
			action      = pred[0]["classes"]                                                          # Pick an action based on the current state
			Q           = pred[0]["probabilities"]
			Q_s_a       = pred[0]["probabilities"][action]
			Q_s_a_n     = pred[0]["probabilities"][action]
			
			if exploration_factor>0.02:
				exploration_factor= 0.9 * exploration_factor #0.98
			else:
				exploration_factor=0.02
			for i in range(1500):
				#with probability e, selects action
				if np.random.random()< exploration_factor:
					action= sample(action_space, 1)[0]                                                   #select select random action
					#print("random action ", action)
					
				#with probability 1-e, selects action=argmax Q(s,a)
				else:
					action      = pred[0]["classes"]                                                    # Pick an action based on the current state
					#print("greedy action ", action)
				
				
				Q           = pred[0]["probabilities"]
				Q_s_a       = pred[0]["probabilities"][action]
				#action      = qlearn.chooseAction(state)
				
				#execute action
				observation, reward, done, info = env.step(action)                                  # Execute the action and get feedback
				#prepare s(t+1)=state2
				img_tmp     = cv2.resize(info, (32, 32), interpolation=cv2.INTER_NEAREST)
				img_tmp     = tf.image.convert_image_dtype(img_tmp, dtype=tf.float32)
				state2      = tf.reshape(img_tmp, shape=(1, 1024*4))
				#predict Q( s(t+1), _)
				predict_input_fn = tf.estimator.inputs.numpy_input_fn(
					x           = {"x": np.array(state2.eval()), "q": np.array([[0, 0, 0]])},      # q does not affect to NN
					batch_size  = 1,
					num_epochs  = 1,
					shuffle     = False)
				
				pred        = list(turtle_regressor.predict(input_fn=predict_input_fn))
				
				action_n    = pred[0]["classes"]  # Pick an action based on the current state
				Q_nxt       = pred[0]["probabilities"]
				Q_s_a_n     = pred[0]["probabilities"][action_n]
				print(i," Q ",Q_nxt,"a ",action_n,"r ",reward)
				#set y_j
				if done:
					y_j   = reward
				else:
					y_j   = reward + gamma * Q_s_a_n
				
				
				x_tmp             = sess.run(state)
				q_tmp             = [Q]
				q_tmp[0][action]  = y_j


#				train_input_fn    = tf.estimator.inputs.numpy_input_fn(
#			            x           = {"x": np.array(x_tmp[0:1]), "q": np.array(q_tmp)},
#             		      y           = np.array(q_tmp[0:1]),
#            		      batch_size  = 1,
#            		      num_epochs  = None,
#            		      shuffle     = False)
#				turtle_regressor.train(
#					input_fn    = train_input_fn,
#					steps       = 10,
#					hooks       = [logging_hook])
			
				
				
				cumulated_reward  += reward

				if highest_reward < cumulated_reward:
					highest_reward    = cumulated_reward



				env._flush(force=True)

				if not(done):
					state             = state2
					Q                 = Q_nxt
				else:
					last_time_steps   = np.append(last_time_steps, [int(i + 1)])
					break
			
			#if x%10==0:
			plotter.plot(env)
			m, s = divmod(int(time.time() - start_time), 60)
			h, m = divmod(m, 60)
			print ("EP: ",str(x+1)," R: ",str(cumulated_reward)," Alpha: ",exploration_factor," Time: %d:%02d:%02d" % (h, m, s))

		#Github table content
		#print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")
	
		l = last_time_steps.tolist()
		l.sort()

		#print("Parameters: a="+str)
		print("Overall score: {:0.2f}".format(last_time_steps.mean()))
		print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

env.close()
