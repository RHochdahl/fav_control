#!/usr/bin/env python

PACKAGE = 'fav_control'
import roslib;roslib.load_manifest(PACKAGE)
import rospy

from dynamic_reconfigure.server import Server
from fav_control.cfg import XControlConfig

import threading
import math
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from depth_controller.msg import StateVector2D
from depth_controller.msg import StateVector3D
from depth_controller.msg import ParametersList


class ControllerNode():
    def __init__(self):
        self.e1 = None
        self.e2 = 0.0
        
        self.data_lock = threading.RLock()

        # 0 =integral-SMC, 1=PID-Controller, 2=SMC with separate i
        self.controller_type = 0

        # PD-Controller, k_d / k_p ~= 0.6
        self.k_p = 1.0
        self.k_d = 0.6

        # SMC
        self.alpha = 0.3
        self.Lambda = 1.5
        self.kappa = 2.5
        self.epsilon = 0.4

        self.int_sat = 0.01

        self.desired_x_pos = -0.5
        self.desired_x_velocity = 0.0
        self.desired_x_acceleration = 0.0
        self.current_x_pos = None
        self.current_x_velocity = None

        self.state_msg_time = 0.0

        self.max_msg_timeout = 0.1

        self.min_x_limit = 0.1
        self.max_x_limit = 1.5

        rospy.init_node("yController")
        
        self.thrust_pub = rospy.Publisher("thrust",
                                                    Float64,
                                                    queue_size=1)
        self.error_pub = rospy.Publisher("x_control_error",
                                          StateVector2D,
                                          queue_size=1)
        self.controller_ready_pub = rospy.Publisher("x_controller_ready",
                                          Bool,
                                          queue_size=1)
        self.state_sub = rospy.Subscriber("estimated_state",
                                          Odometry,
                                          self.get_current_state,
                                          queue_size=1)

        self.time = rospy.get_time()

        rospy.sleep(5.0)
        self.report_readiness(True)

        self.server = Server(YControlConfig, self.server_callback)

        self.setpoint_sub = rospy.Subscriber("x_setpoint",
                                            StateVector3D,
                                            self.get_setpoint,
                                            queue_size=1)

    def send_control_message(self, u):
        msg = Float64()
        msg.data = u
        self.thrust_pub.publish(msg)

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
            rospy.loginfo("New Parameters received by x_Controller")

            self.controller_type = config.controller_type

            self.k_p = config.k_p
            self.k_d = config.k_d

            self.alpha = config.alpha
            self.Lambda = config.Lambda
            self.kappa = config.kappa
            self.epsilon = config.epsilon
            
        return config

    def run(self):
        rate = rospy.Rate(10.0)

        while not rospy.is_shutdown():
            u = self.controller()
            self.send_control_message(u)

            self.publish_error()

            rate.sleep()
        
    def get_setpoint(self, msg):
        with self.data_lock:
            self.desired_x_pos = msg.x_position
            self.desired_x_velocity = msg.x_velocity
            self.desired_x_acceleration = msg.x_acceleration
    
    def get_current_state(self, msg):
        with self.data_lock:
            self.current_x_pos = msg.pose.pose.position.x
            self.current_x_velocity = msg.twist.twist.linear.x
            self.state_msg_time = rospy.get_time()

    def controller(self):
        if (rospy.get_time() - self.state_msg_time > self.max_msg_timeout):
            rospy.logwarn_throttle(1.0, "No state information received!")
            return 0.0

        if self.controller_type is None:
            rospy.logwarn_throttle(1.0, "No controller chosen!")
            return 0.0

        # return 0.0 if x_pos or setpoint is 'unsafe'
        if ((self.desired_x_pos < self.min_x_limit) or (self.desired_x_pos > self.max_x_limit)):
            rospy.logwarn_throttle(1.0, "x setpoint outside safe region!")
            return 0.0

        # return 0.0 if x_pos or setpoint is 'unsafe'
        if ((self.current_x_pos < self.min_x_limit) or (self.current_x_pos > self.max_x_limit)):
            rospy.logwarn_throttle(5.0, "Diving x outside safe region!")
            return 0.0
        
        delta_t = rospy.get_time() - self.time
        self.time = rospy.get_time()
        
        if self.controller_type == 0:
            # integral-SMC
            self.e1 = self.desired_x_pos - self.current_x_pos
            self.e2 = self.desired_x_velocity - self.current_x_velocity
            s = self.e2 + self.Lambda*self.e1
            u = self.alpha*(self.desired_x_acceleration+self.Lambda*self.e2+self.kappa*(s/(abs(s)+self.epsilon)))

        elif self.controller_type == 1:
            # PID-Controller
            self.e1 = self.desired_x_pos - self.current_x_pos
            self.e2 = self.desired_x_velocity - self.current_x_velocity
            u = self.k_p * self.e1 + self.k_d * self.e2
            
        else:
            rospy.logerr_throttle(10.0, "\nError! Undefined Controller chosen.\n")
            return 0.0

        return self.sat(u)

    def sat(self, x, limit=1.0):
        return min(max(x, -limit), limit)

def main():
   node = ControllerNode()
   node.run()


if __name__ == "__main__":
   main()
