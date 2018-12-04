import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2,cv_bridge
import math
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from gym.utils import seeding

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

class GazeboCircuit2TurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "mycolor.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(3) #F,L,R
        self.reward_range = (-np.inf, np.inf)

        self._seed()
        cv2.namedWindow("window", 1)

        self.initial_odom= rospy.wait_for_message('/odom', Odometry, timeout=5)
        print("initial", self.initial_odom)
        self.prev_pos_x=0.0
        self.prev_pos_y=0.0
        self.curr_pos_x=0.0
        self.curr_pos_y=0.0

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: #FORWARD
            #print("Forward")
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            #print("Left")
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.5
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            #print("Right")
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.5
            self.vel_pub.publish(vel_cmd)

        data = None
        img_data=None
        image=None
        odom=None
        while data is None:
            try:

                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                odom = rospy.wait_for_message('/odom', Odometry, timeout=5)
                img_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                #h = image_data.height
                #w = image_data.width
                #turn kinect image to cv2 image
                
                im_cl=cv_bridge.CvBridge().imgmsg_to_cv2(img_data,desired_encoding='bgra8')
                #print("image_shape",im_cl.shape)
                #cv2.imshow("window",im_cl)
                #cv2.waitKey(1)
            
                self.prev_pos_x=self.curr_pos_x
                self.prev_pos_y=self.curr_pos_y
                self.curr_pos_x=odom.pose.pose.position.x-self.initial_odom.pose.pose.position.x
                self.curr_pos_y=odom.pose.pose.position.y-self.initial_odom.pose.pose.position.y
                
            except:
                pass
                print("________NOTHING_______")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(data,5)
        #print("{0:.2f}".format(self.curr_pos_x),"{0:.2f}".format(self.curr_pos_y))
        #print("v","{0:.2f}".format(odom.pose.pose.position.x),"{0:.2f}".format(odom.pose.pose.position.y))
        reward=0
        # mask red on image
        boundaries = [
		([5, 5, 80], [90, 70, 250])
        ]

        image=im_cl[:,:,0:3]
        for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
 
		# find the colors within the specified boundaries and apply
		# the mask
		#print(lower.shape,image.shape)
		mask = cv2.inRange(image, lower, upper)
		output = cv2.bitwise_and(image, image, mask = mask)
		#print("sum% ",np.sum(output)/output.size)
		# show the images
		cv2.imshow("window", image)#np.hstack([image, output]))
		cv2.waitKey(1)

        if not done:
            #print("not done")
            if (np.sum(output)/output.size >8):	#(abs(self.curr_pos_x - (-5))<2 and abs(self.curr_pos_y - (5))<2):
            	reward=5000
            	print("*************************** YOU WON ******************************")
            	done=1
            else:
            	if (abs(self.curr_pos_y - (self.prev_pos_y))>0.1 or abs(self.curr_pos_x - (self.prev_pos_x))>0.1):
            		reward = 5 * np.sum(output)/output.size
            		#print("** KEEP MOVING **")    	
        else:
            reward = -500

        return state, reward, done, im_cl

    def _reset(self):
        
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #read laser data
        data = None
        img_data= None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                img_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                image = cv_bridge.CvBridge().imgmsg_to_cv2(img_data,desired_encoding='mono8')
                im_cl=cv_bridge.CvBridge().imgmsg_to_cv2(img_data,desired_encoding='bgra8')
                #turn grayscale image to binary image(inorder to have more contrast)
                #(thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                self.initial_odom= rospy.wait_for_message('/odom', Odometry, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        
        state = self.discretize_observation(data,5)

        return state ,im_cl


    #def _initialPosition(self):
    #    self.initial_odom= rospy.wait_for_message('/odom', Odometry, timeout=5)
    #    print(self.initial_odom.pose.pose.position)
