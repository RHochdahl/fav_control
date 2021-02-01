#!/usr/bin/env python

PACKAGE = 'fav_control'
import roslib;roslib.load_manifest(PACKAGE)
import rospy

from dynamic_reconfigure.server import Server
from fav_control.cfg import YawControlConfig

import tf
import numpy as np

import threading
import math
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistWithCovarianceStamped
from fav_control.msg import StateVector2D
from fav_control.msg import StateVector3D


class ControllerNode():
    def __init__(self):
        self.simulate = rospy.get_param("simulate")
        self.use_ground_truth = rospy.get_param("use_ground_truth")

        self.e1 = 0.0
        self.e2 = 0.0
        
        self.data_lock = threading.RLock()

        # 0 =SMC, 1=PD-Controller
        self.controller_type = 0

        # PD-Controller, k_d / k_p ~= 0.6
        self.k_p = 3.0
        self.k_d = 1.4

        # SMC
        self.alpha = 0.3
        self.Lambda = 1.0
        self.kappa = 1.2
        self.epsilon = 0.7

        self.desired_yaw = -0.5
        self.desired_yaw_vel = 0.0
        self.desired_yaw_acc = 0.0
        self.current_yaw = None
        self.current_yaw_vel = None
        self.yaw_uncertainty = 1.0
        self.k_u = 2.0

        self.pose_msg_time = 0.0
        self.twist_msg_time = 0.0

        self.max_msg_timeout = 0.1

        self.yaw_d_limit = 1.0

        rospy.init_node("yawController")
        
        self.yaw_pub = rospy.Publisher("yaw",
                                        Float64,
                                        queue_size=1)
        self.error_pub = rospy.Publisher("yaw_control_error",
                                          StateVector2D,
                                          queue_size=1)
        self.controller_ready_pub = rospy.Publisher("yaw_controller_ready",
                                          Bool,
                                          queue_size=1)

        self.pose_sub = rospy.Subscriber("ekf_pose",
                                         PoseWithCovarianceStamped,
                                         self.get_current_pose,
                                         queue_size=1)
        self.twist_sub = rospy.Subscriber("ekf_twist",
                                          TwistWithCovarianceStamped,
                                          self.get_current_twist,
                                          queue_size=1)

        rospy.sleep(5.0)
        self.report_readiness(True)

        self.server = Server(YawControlConfig, self.server_callback)

        self.setpoint_sub = rospy.Subscriber("yaw_setpoint",
                                            StateVector3D,
                                            self.get_setpoint,
                                            queue_size=1)

    def send_control_message(self, u):
        msg = Float64()
        msg.data = u
        self.yaw_pub.publish(msg)

    def publish_error(self):
        err_msg = StateVector2D()
        err_msg.header.stamp = rospy.Time.now()
        err_msg.position = self.e1
        err_msg.velocity = self.e2
        self.error_pub.publish(err_msg)

    def report_readiness(self, ready_bool):
        msg = Bool()
        msg.data = ready_bool
        self.controller_ready_pub.publish(msg)

    def server_callback(self, config, level):
        with self.data_lock:
            if config.dynamic_reconfigure:
                rospy.loginfo("New Parameters received by yaw_Controller")

                self.controller_type = config.controller_type
                self.k_u = config.k_u

                self.k_p = config.k_p
                self.k_d = config.k_d

                self.alpha = config.alpha
                self.Lambda = config.Lambda
                self.kappa = config.kappa
                self.epsilon = config.epsilon

            else:
                config.controller_type = self.controller_type
                config.k_u = self.k_u

                config.k_p = self.k_p
                config.k_d = self.k_d

                config.alpha = self.alpha
                config.Lambda = self.Lambda
                config.kappa = self.kappa
                config.epsilon = self.epsilon
            
        return config

    def run(self):
        rate = rospy.Rate(50.0)

        while not rospy.is_shutdown():
            u = self.controller()
            self.send_control_message(u)

            self.publish_error()

            rate.sleep()
        
    def get_setpoint(self, msg):
        with self.data_lock:
            self.desired_yaw = msg.position
            self.desired_yaw_vel = msg.velocity
            self.desired_yaw_acc = msg.acceleration

    def get_current_pose(self, msg):
        with self.data_lock:
            quaternion = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            self.current_yaw = tf.transformations.euler_from_quaternion(quaternion)[2]
            self.yaw_uncertainty = msg.pose.covariance[35]
            self.pose_msg_time = rospy.get_time()

    def get_current_twist(self, msg):
        with self.data_lock:
            self.current_yaw_vel = msg.twist.twist.angular.z
            self.twist_msg_time = rospy.get_time()

    def controller(self):
        if (rospy.get_time() - self.pose_msg_time > self.max_msg_timeout):
            rospy.logwarn_throttle(10.0, "No pose received!")
            return 0.0

        if (rospy.get_time() - self.twist_msg_time > self.max_msg_timeout):
            rospy.logwarn_throttle(10.0, "No twist received!")
            return 0.0

        if self.controller_type is None:
            rospy.logwarn_throttle(10.0, "No controller chosen!")
            return 0.0
        
        # return 0.0 if setpoint is 'unsafe'
        if ((self.desired_yaw_vel < -self.yaw_d_limit) or (self.desired_yaw_vel > self.yaw_d_limit)):
            rospy.logwarn_throttle(10.0, "yaw angular velocity setpoint outside safe region!")
            return 0.0

        if self.yaw_uncertainty > 1.0:
            return 0

        if self.controller_type == 0:
            # SMC
            self.e1 = self.get_angular_error(self.desired_yaw, self.current_yaw)
            self.e2 = self.desired_yaw_vel - self.current_yaw_vel
            s = self.e2 + self.Lambda*self.e1
            u = self.alpha*(self.desired_yaw_acc+self.Lambda*self.e2+self.kappa*(s/(abs(s)+self.epsilon)))

        elif self.controller_type == 1:
            # PD-Controller
            self.e1 = self.get_angular_error(self.desired_yaw, self.current_yaw)
            self.e2 = self.desired_yaw_vel - self.current_yaw_vel
            u = self.k_p * self.e1 + self.k_d * self.e2

        else:
            rospy.logerr_throttle(10.0, "\nError! Undefined Controller chosen.\n")
            return 0.0

        k = 10**(-self.k_u*self.yaw_uncertainty)
        return self.sat(k*u)

    def get_angular_error(self, desired_angle, current_angle):
        e = (desired_angle % (2*np.pi)) - (current_angle % (2*np.pi))
        if e > np.pi:
            return e - 2*np.pi
        elif e < -np.pi:
            return e + 2*np.pi
        else:
            return e

    def sat(self, x, limit=1.0):
        return min(max(x, -limit), limit)


def main():
    node = ControllerNode()
    node.run()


if __name__ == "__main__":
    main()
