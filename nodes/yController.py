#!/usr/bin/env python

PACKAGE = 'fav_control'
import roslib;roslib.load_manifest(PACKAGE)
import rospy

from dynamic_reconfigure.server import Server
from fav_control.cfg import YControlConfig

import threading
import math
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from fav_control.msg import StateVector2D
from fav_control.msg import StateVector3D


class ControllerNode():
    def __init__(self):
        self.simulate = rospy.get_param("simulate")
        self.use_ground_truth = rospy.get_param("use_ground_truth")

        self.e1 = 0.0
        self.e2 = 0.0
        
        self.data_lock = threading.RLock()

        # 0 =integral-SMC, 1=PID-Controller, 2=SMC with separate i
        self.controller_type = 1

        # PD-Controller, k_d / k_p ~= 0.6
        self.k_p = 2.0
        self.k_d = 0.6

        # SMC
        self.alpha = 0.5
        self.Lambda = 1.5
        self.kappa = 2.5
        self.epsilon = 0.4

        self.desired_y_pos = -0.5
        self.desired_y_velocity = 0.0
        self.desired_y_acceleration = 0.0
        self.current_y_pos = None
        self.current_y_velocity = None
        self.y_uncertainty = 1.0
        self.k_u = 2.0

        self.state_msg_time = 0.0

        self.max_msg_timeout = 0.1

        self.min_y_limit = 0.1
        self.max_y_limit = 3.25
        self.y_d_limit = 0.5

        rospy.init_node("yController")
        
        self.lateral_thrust_pub = rospy.Publisher("lateral_thrust",
                                                    Float64,
                                                    queue_size=1)
        self.error_pub = rospy.Publisher("y_control_error",
                                          StateVector2D,
                                          queue_size=1)
        self.controller_ready_pub = rospy.Publisher("y_controller_ready",
                                          Bool,
                                          queue_size=1)
        if self.use_ground_truth and self.simulate:
            self.state_sub = rospy.Subscriber("/ground_truth/state",
                                            Odometry,
                                            self.get_current_state,
                                            queue_size=1)
        else:
            self.state_sub = rospy.Subscriber("estimated_state",
                                            Odometry,
                                            self.get_current_state,
                                            queue_size=1)

        rospy.sleep(5.0)
        self.report_readiness(True)

        self.server = Server(YControlConfig, self.server_callback)

        self.setpoint_sub = rospy.Subscriber("y_setpoint",
                                            StateVector3D,
                                            self.get_setpoint,
                                            queue_size=1)
    
    def run(self):
        rate = rospy.Rate(50.0)

        while not rospy.is_shutdown():
            u = self.controller()
            self.send_control_message(u)

            self.publish_error()

            rate.sleep()
        
    def send_control_message(self, u):
        msg = Float64()
        msg.data = u
        self.lateral_thrust_pub.publish(msg)

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
            rospy.loginfo("New Parameters received by y_Controller")

            self.controller_type = config.controller_type
            self.k_u = config.k_u

            self.k_p = config.k_p
            self.k_d = config.k_d

            self.alpha = config.alpha
            self.Lambda = config.Lambda
            self.kappa = config.kappa
            self.epsilon = config.epsilon
            
        return config

    def get_setpoint(self, msg):
        with self.data_lock:
            self.desired_y_pos = msg.position
            self.desired_y_velocity = msg.velocity
            self.desired_y_acceleration = msg.acceleration
    
    def get_current_state(self, msg):
        with self.data_lock:
            self.current_y_pos = msg.pose.pose.position.y
            self.current_y_velocity = msg.twist.twist.linear.y
            self.y_uncertainty = msg.pose.covariance[7]
            self.state_msg_time = rospy.get_time()

    def controller(self):
        if (rospy.get_time() - self.state_msg_time > self.max_msg_timeout):
            rospy.logwarn_throttle(10.0, "No state information received!")
            return 0.0

        if self.controller_type is None:
            rospy.logwarn_throttle(10.0, "No controller chosen!")
            return 0.0

        # return 0.0 if setpoint is 'unsafe'
        if ((self.desired_y_pos < self.min_y_limit) or (self.desired_y_pos > self.max_y_limit)):
            rospy.logwarn_throttle(10.0, "y setpoint outside safe region!")
            return 0.0

        # return 0.0 if setpoint velocity is 'unsafe'
        if ((self.desired_y_velocity < -self.y_d_limit) or (self.desired_y_velocity > self.y_d_limit)):
            rospy.logwarn_throttle(10.0, "y velocity setpoint outside safe region!")
            return 0.0

        # return 0.0 if y_pos is 'unsafe'
        if ((self.current_y_pos < self.min_y_limit) or (self.current_y_pos > self.max_y_limit)):
            rospy.logwarn_throttle(10.0, "y outside safe region!")
            return 0.0
        
        if self.controller_type == 0:
            # integral-SMC
            self.e1 = self.desired_y_pos - self.current_y_pos
            self.e2 = self.desired_y_velocity - self.current_y_velocity
            s = self.e2 + self.Lambda*self.e1
            k = 10**(-self.k_u*self.y_uncertainty)
            u = k * self.alpha*(self.desired_y_acceleration+self.Lambda*self.e2+self.kappa*(s/(abs(s)+self.epsilon)))

        elif self.controller_type == 1:
            # PID-Controller
            self.e1 = self.desired_y_pos - self.current_y_pos
            self.e2 = self.desired_y_velocity - self.current_y_velocity
            k = 10**(-self.k_u*self.y_uncertainty)
            u = k * (self.k_p * self.e1 + self.k_d * self.e2)
            
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
