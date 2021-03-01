#!/usr/bin/env python
import gym
import numpy as np
from gym import spaces
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
import rospy
import time
import math
from std_msgs.msg import Bool, Float32, Float64, Header
from ackermann_msgs.msg import AckermannDriveStamped
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

class DeepracerEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):

        super(DeepracerEnv, self).__init__()

        self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)
        self.msg = AckermannDriveStamped()
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]),
                                        dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-10,-10,-10]), high=np.array([10,10,10]),
                                        dtype=np.float32)

        self.state = [0,0,0]

    def state_callback(self, data):
        racecar_pose = data.pose[-1]
        self.state[0] = racecar_pose.position.x
        self.state[1] = racecar_pose.position.y
        quaternion = (
            data.pose[-1].orientation.x,
            data.pose[-1].orientation.y,
            data.pose[-1].orientation.z,
            data.pose[-1].orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.state[2] = euler[2]

    def step(self, action):
        self.steering_angle = action[0]
        self.speed = action[1]
        self.send_action()      

        self.state = [0,0,0]
        self.reward = None
        self.done = False

        return np.array(self.state).astype(np.float32), self.reward, self.done, {}

    def reset(self):
        self.reward = None
        self.done = False

        self.steering_angle = 0
        self.speed = 0
        self.send_action()
        
        print("Reset")

        return np.array(self.agent_pos).astype(np.float32)  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close (self):
        pass

    def send_action(self):
        self.msg.drive.steering_angle = self.steering_angle
        self.msg.drive.speed = self.speed
        self.pub.publish(self.msg)



if __name__ == '__main__':
    try:
        env = DeepracerEnv()     
        
        print('===================Check===============', check_env(env))

    except rospy.ROSInterruptException:
        pass






