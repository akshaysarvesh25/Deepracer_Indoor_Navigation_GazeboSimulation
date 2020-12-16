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

flag_move = 0
x_des = 10
y_des = 0
x_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output',AckermannDriveStamped,queue_size=1)
throttle = 0.0
pos=[0,0]
def set_throttle_steer(data):
    #print(data.pose[1].orientation.y)
    racecar_pose = data.pose[1]
    pos[0] = racecar_pose.position.x
    pos[1] = racecar_pose.position.y    


def servo_commands():

    #time.sleep(2)
    rospy.init_node('servo_commands', anonymous=True)
    
    msg = AckermannDriveStamped()
    #rospy.Subscriber("/progress", Progress, set_throttle_steer)
    rospy.Subscriber("/gazebo/model_states", ModelStates,set_throttle_steer)
    #print("========first throttle signal=======",throttle) 
    err = math.sqrt((x_des-pos[0])**2+(y_des-pos[1])**2)
    heading = math.atan((y_des-pos[1])/(x_des-pos[0]))
    print("=====heading=====",heading)
    while not rospy.is_shutdown:
        #msg.drive.speed = 5.0
        msg.drive.steering = 10.0
        x_pub.publish(msg)
        time.sleep(1)
        heading = math.atan((y_des-pos[1])/(x_des-pos[0]))
        print("=====heading=====",heading)

        """

        #msg.drive.speed = 5.0
        msg.drive.steering = 10.0
        x_pub.publish(msg)
        time.sleep(1)

        #msg.drive.speed = 5.0
        msg.drive.steering = 10.0
        x_pub.publish(msg)
        time.sleep(1)

        #msg.drive.speed = 5.0
        msg.drive.steering = 10.0
        x_pub.publish(msg)
        time.sleep(1)

        #msg.drive.speed = 5.0
        msg.drive.steering = 10.0
        x_pub.publish(msg)
        time.sleep(1)
        """
        print("new loop")
    
    """

    while not rospy.is_shutdown:
        msg.drive.speed = 1.0
    	x_pub.publish(msg)
	time.sleep(1)
    """

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        servo_commands()
    except rospy.ROSInterruptException:
        pass
