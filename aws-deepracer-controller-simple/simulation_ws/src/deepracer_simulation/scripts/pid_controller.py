#!/usr/bin/env python
import rospy
import time
import math
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
from gazebo_msgs.msg import ModelStates
import PID_control
import tf
import pandas as pd 
import numpy as np

# Publish to the Ackermann Control Topic
x_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)

# Initiate variables
throttle = 0.0
heading = 0.0
pos=[0,0]
yaw=0.0

# Setup a waypoint counter
global count
count = 0

    
def set_position(data):
    ### Subscribe from topic and parse the pose data for robot ###

    global x_des
    global y_des

    ### To check for single goal point ###
    ### Uncomment these lines and comment lines 114-117 ###
    # x_des = 3
    # y_des = 0

    ### Parsing the data ###
    racecar_pose = data.pose[-1]   
        
    pos[0] = racecar_pose.position.x
    pos[1] = racecar_pose.position.y
    quaternion = (
            data.pose[-1].orientation.x,
            data.pose[-1].orientation.y,
            data.pose[-1].orientation.z,
            data.pose[-1].orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]  

    ### Previous Error and Heading ###
    prev_err = math.sqrt((x_des-pos[0])**2+(y_des-pos[1])**2)
    prev_head = math.atan((y_des-pos[1])/(x_des-pos[0]+0.00001))

    ### PID control of car if error distance < 0.5 else stop the car and wait for next waypoint ###
    if (prev_err<=0.5):
        stop_car()
        sub.unregister()        
        servo_commands()        
    else:
        t1 = time.time()
        control_car(t1, pos,yaw, prev_err, prev_head)        
        

def control_car(t1,pos,yaw, prev_err, prev_head):
    
    msg = AckermannDriveStamped()

    ### PID for throttle control ###
    speed_control = PID_control.PID(0.5,0,0.8)
    err = math.sqrt((x_des-pos[0])**2+(y_des-pos[1])**2)
    dt = time.time() - t1
    throttle = speed_control.Update(dt, prev_err, err)

    ### PID for steer control ###   
    steer_control = PID_control.PID(0.5,0,0)
    head = math.atan((y_des-pos[1])/(x_des-pos[0]+0.01))
    steer = steer_control.Update(dt, prev_head-yaw, head - yaw)

    ### Publish the control signals to Ackermann control topic ###
    msg.drive.speed = throttle 
    msg.drive.steering_angle = steer
    x_pub.publish(msg)

    # Store previous error
    prev_err = err 

def stop_car():
    ### Stop the car as is ###
    msg = AckermannDriveStamped()
    msg.drive.speed = 0
    msg.drive.steering_angle = 0
    msg.drive.steering_angle_velocity = 0
    x_pub.publish(msg)
    print("Goal Reached!") 

def servo_commands():


    ######### For user-input waypoints ########
    # print("Car is at :",pos[0],pos[1])
    # print("Enter Waypoints:")
    # x_des = float(input())
    # y_des = float(input())
    
    global x_des
    global y_des
    global sub
    global count

    ### For parsing waypoints from a .csv file ###    
    x_des = (x[count])
    y_des = (y[count])
    print('Navigating to: ',x_des, y_des)
    count +=1
     
    #msg = AckermannDriveStamped()

    ### Subscribe to /gazebo/model_states for pose data feedback ###
    sub = rospy.Subscriber("/gazebo/model_states", ModelStates, set_position)   

    while not (rospy.is_shutdown()):
        '''
        '''      

    time.sleep(0.1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('servo_commands', anonymous=True)

        ### For parsing waypoints from a .csv file ###       
        global df
        global x, y
        df = pd.read_csv('route_smooth.csv')
        x = np.array(df["X"])
        y = np.array(df["Y"])  

        servo_commands()

    except rospy.ROSInterruptException:
        stop_car()
        sub.unregister()
        pass

