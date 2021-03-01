#!/usr/bin/env python
import rospy

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Pose2D
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler


path = Path()

def set_position_1(data):
    header = Header()
    header.frame_id='/map'
    header.stamp = rospy.Time.now()
    global path
    racecar_pose = data.pose[-1]
    path.header = header
    pose = PoseStamped()
    pose.header = path.header
    pose.pose = racecar_pose
    path.poses.append(pose)
    path_pub.publish(path)
    #print("published")

def set_position(data):
    header = Header()
    header.frame_id='/base_link'
    header.stamp = rospy.Time.now()
    global path
    
    #racecar_pose = data.pose[-1]
    path.header = header
    pose = PoseStamped()
    pose.header = path.header
    pose.pose.position.x = data.x
    pose.pose.position.y = data.y
    pose.pose.position.z = 0.0

    q = quaternion_from_euler(0, 0, -data.theta)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]  

    pose_pub.publish(pose)

    path.poses.append(pose)
    path_pub.publish(path)
    #print("published")

rospy.init_node('path_node')

#odom_sub = rospy.Subscriber('/odom', Odometry, odom_cb)
#rospy.Subscriber("/gazebo/model_states", ModelStates, set_position)
rospy.Subscriber("/pose2D", Pose2D, set_position)
path_pub = rospy.Publisher('/path', Path, queue_size=10)
pose_pub = rospy.Publisher('/pose', PoseStamped, queue_size=10)

if __name__ == '__main__':
    rospy.spin()
