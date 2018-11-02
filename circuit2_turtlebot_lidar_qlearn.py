#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy 
import random
import time
import cv2
import tensorflow as tf
from skimage import transform 

import qlearn
import liveplot



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

    last_time_steps = numpy.ndarray(0)

    #qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    #alpha=0.2, gamma=0.8, epsilon=0.9)

    #initial_epsilon = qlearn.epsilon

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

    with tf.Session() as sess:
    	sess.run(init)
    	for x in range(total_episodes):
        	done = False
        	cumulated_reward = 0 
        	observation,info = env.reset()
        	image = cv2.resize(info, (32,32), interpolation = cv2.INTER_NEAREST)                       		
        	image0 = tf.image.convert_image_dtype(image, dtype=tf.float32)
        	image1  = tf.reshape(image0, shape=(1,1024))

        	#if qlearn.epsilon > 0.05:
            	#	qlearn.epsilon *= epsilon_discount

        	#render() #defined above, not env.render()
        	state = ''.join(map(str, observation))
        	#print("state:-------------",state);
        	for i in range(1500):
            		# Pick an action based on the current state
            		action,allQ = sess.run([predict,Qout],feed_dict={inputs1:image1.eval()})
            		# Execute the action and get feedback
            		#print("action------------",action)
            		observation, reward, done, info = env.step(action)

            		image = cv2.resize(info, (32,32), interpolation = cv2.INTER_NEAREST)                        	
            		image0 = tf.image.convert_image_dtype(image, dtype=tf.float32)
            		image2  = tf.reshape(image0, shape=(1,1024))

            		Q1 = sess.run(Qout,feed_dict={inputs1:image2.eval()})

            		maxQ1 = numpy.max(Q1)
            		targetQ = allQ
            		targetQ[0,action] = reward + 0.97*maxQ1
            		
            		#Train our network using target and predicted Q values
            		_,W1 = sess.run([updateModel,W],feed_dict={inputs1:image1.eval(),nextQ:targetQ})
            		image1=image2

            		cumulated_reward += reward
            		if highest_reward < cumulated_reward:
                		highest_reward = cumulated_reward
			
            		nextState = ''.join(map(str, observation))
            		
            		env._flush(force=True)
            		if not(done):
                		state = nextState
            		else:
                		last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
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
