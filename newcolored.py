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

import liveplot
from dqn_agent import DQNAgent
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#--------------------------------------------------------------------------------------------------------------------------------------
def render():
    render_skip       = 0 #Skip first X episodes.
    render_interval   = 50 #Show render Every Y episodes.
    render_episodes   = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)
        
#--------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #------------------------------------------------------------------------
    env               = gym.make('GazeboCircuit2TurtlebotLidar-v0')
    outdir            = '/tmp/gazebo_gym_experiments'
    env               = gym.wrappers.Monitor(env, outdir, force=True)
    plotter           = liveplot.LivePlot(outdir)
    last_time_steps   = np.ndarray(0)
    start_time        = time.time()
    total_episodes    = 1000
    max_steps         = 300
    highest_reward    = 0
    gamma             = 0.95
    num_actions       = 3
    action_space      = [0,1,2]
    tf.reset_default_graph()                             # Reset training graph                                   
    myinit            = tf.global_variables_initializer()# Initialize training network 
    
    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)
    #------------------------------------------------------------------------
    agent=DQNAgent(action_space,"GazeboCircuit2TurtlebotLidar-v0")
    
    agent.exploration=1
    cv2.namedWindow("window", 1)
    x_val = np.random.rand(4096,4096).astype(np.float32)
    agent.W_fc1.load(x_val, session=agent.sess)
    
    for e in range(total_episodes):
        # reset
        terminal= False
        win     = 0
        frame   = 0
        loss    = 0.0
        Q_max   = 0.0
        steps   = 0
        reward_t= 0.0
        env.reset()
        cumulated_rewards  = 0
        agent.exploration *= 0.9
        if agent.exploration<0.1:
            agent.exploration=0.1
            
        _, reward, terminal, info = env.step(0)
            
        img_tmp     = cv2.resize(info, (32, 32), interpolation=cv2.INTER_NEAREST)
        state_t_1   = tf.image.convert_image_dtype(img_tmp, dtype=tf.float32)
        state_t_1   = tf.reshape(state_t_1,(-1,32,32,4))
        
        
        while (not terminal):
            steps  += 1
            state_t = state_t_1

            # execute action in environment
            action_t = agent.select_action(state_t, agent.exploration)
            _, reward_t, terminal, info = env.step(action_t)
            #print("step: ", steps, "action: ",action_t ,"reward: ", reward_t)
            print(action_t , end="")
            img_tmp     = cv2.resize(info, (32, 32), interpolation=cv2.INTER_NEAREST)
            state_t_1   = tf.image.convert_image_dtype(img_tmp, dtype=tf.float32)
            state_t_1   = tf.reshape(state_t_1,(-1,32,32,4))
            # store experience
            agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)
            # experience replay
            agent.experience_replay()
            #print(agent.sess.run(agent.W_fc1))
            
            # for log
            frame += 1
            loss  += agent.current_loss
            Q_max += np.max(agent.Q_values(state_t))
            cumulated_rewards += reward_t
        
        
        
        print(" ")
        print("episodes:",e," steps:",steps," loss:",'{0:.2f}'.format(loss/(steps+1)), " terminal:",terminal, " exploration_factor:",agent.exploration , " reward:", '{0:.2f}'.format(cumulated_rewards))
        plotter.plot(env)
        #print("EPOCH: {:03d}/{:03d} | WIN: {:03d} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
        #    e, total_episodes - 1, cumulated_rewards, loss / frame, Q_max / frame))
        env._flush(force=True)
        # save model
        if e%50==49:
            weights_tmp     = cv2.resize(agent.sess.run(agent.W_fc1), (500, 500), interpolation=cv2.INTER_NEAREST)
            weights_image   = tf.image.convert_image_dtype(weights_tmp, dtype=tf.float32)
            cv2.imshow("window",agent.sess.run(weights_image))
            cv2.waitKey(1)
            agent.save_model() 
    # save model
    agent.save_model()    
    
    env.close()






        
        
