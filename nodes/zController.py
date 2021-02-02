#!/usr/bin/env python

PACKAGE = 'fav_control'
# import roslib;roslib.load_manifest(PACKAGE)
import rospy

from dynamic_reconfigure.server import Server
from fav_control.cfg import ZControlConfig

import threading
import math
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from fav_control.msg import StateVector2D
from fav_control.msg import StateVector3D


class ControllerNode():
    def __init__(self):
        rospy.init_node("zController")

        self.simulate = rospy.get_param("simulate", True)
        self.use_ground_truth = rospy.get_param("use_ground_truth", False)

        self.e1 = 0.0
        self.e2 = 0.0
        
        self.data_lock = threading.RLock()

        # 0 =integral-SMC, 1=PID-Controller, 2=SMC with separate i
        self.controller_type = 0

        # PD-Controller, k_d / k_p ~= 0.6
        self.k_p = 9.0
        self.k_d = 5.8

        # SMC
        self.alpha = 1.0
        self.Lambda = 1.0
        self.kappa = 1.2
        self.epsilon = 0.7

        self.int_sat = 0.01

        self.desired_z_pos = -0.5
        self.desired_z_velocity = 0.0
        self.desired_z_acceleration = 0.0
        self.current_z_pos = None
        self.current_z_velocity = None
        self.z_uncertainty = 1.0
        self.k_u = 2.0

        self.pose_msg_time = 0.0
        self.twist_msg_time = 0.0

        self.max_msg_timeout = 0.1

        self.deep_z_limit = -0.8
        self.shallow_z_limit = -0.1
        self.z_d_limit = 0.5

        self.k_i = 0.2  # 1.0
        self.integrator_buffer = 0.0

        self.vertical_thrust_pub = rospy.Publisher("vertical_thrust",
                                                   Float64,
                                                   queue_size=1)
        self.error_pub = rospy.Publisher("z_control_error",
                                         StateVector2D,
                                         queue_size=1)
        self.controller_ready_pub = rospy.Publisher("z_controller_ready",
                                                    Bool,
                                                    queue_size=1)

        if self.use_ground_truth and self.simulate:
            self.ground_truth_state_sub = rospy.Subscriber("ground_truth/state",
                                                           Odometry,
                                                           self.get_ground_truth,
                                                           queue_size=1)
        else:
            self.pose_sub = rospy.Subscriber("ekf_pose",
                                             PoseWithCovarianceStamped,
                                             self.get_current_pose,
                                             queue_size=1)
            self.twist_sub = rospy.Subscriber("ekf_twist",
                                              TwistWithCovarianceStamped,
                                              self.get_current_twist,
                                              queue_size=1)

        self.time = rospy.get_time()

        rospy.sleep(5.0)
        self.report_readiness(True)

        self.server = Server(ZControlConfig, self.server_callback)

        self.setpoint_sub = rospy.Subscriber("z_setpoint",
                                             StateVector3D,
                                             self.get_setpoint,
                                             queue_size=1)

    def send_control_message(self, u):
        msg = Float64()
        msg.data = u
        self.vertical_thrust_pub.publish(msg)

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
                rospy.loginfo("New Parameters received by z_Controller")

                self.controller_type = config.controller_type
                self.k_u = config.k_u
                
                if config.reset_integrator:
                    self.integrator_buffer = 0.0
                    config.reset_integrator = False

                self.k_p = config.k_p
                self.k_d = config.k_d

                self.alpha = config.alpha
                self.Lambda = config.Lambda
                self.kappa = config.kappa
                self.epsilon = config.epsilon
                self.int_sat = config.int_sat

            else:
                config.controller_type = self.controller_type
                config.k_u = self.k_u
                
                config.reset_integrator = False

                config.k_p = self.k_p
                config.k_d = self.k_d

                config.alpha = self.alpha
                config.Lambda = self.Lambda
                config.kappa = self.kappa
                config.epsilon = self.epsilon
                config.int_sat = self.int_sat
            
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
            self.desired_z_pos = msg.position
            self.desired_z_velocity = msg.velocity
            self.desired_z_acceleration = msg.acceleration
    
    def get_current_pose(self, msg):
        with self.data_lock:
            self.current_z_pos = msg.pose.pose.position.z
            self.z_uncertainty = msg.pose.covariance[14]
            self.pose_msg_time = rospy.get_time()

    def get_current_twist(self, msg):
        with self.data_lock:
            self.current_z_velocity = msg.twist.twist.linear.z
            self.twist_msg_time = rospy.get_time()

    def get_ground_truth(self, msg):
        with self.data_lock:
            self.current_z_pos = msg.pose.pose.position.z
            self.z_uncertainty = msg.pose.covariance[14]
            self.pose_msg_time = rospy.get_time()
            
            self.current_z_velocity = msg.twist.twist.linear.z
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
        if ((self.desired_z_pos < self.deep_z_limit) or (self.desired_z_pos > self.shallow_z_limit)):
            rospy.logwarn_throttle(10.0, "z setpoint outside safe region!")
            return 0.0

        # return 0.0 if setpoint velocity is 'unsafe'
        if ((self.desired_z_velocity < -self.z_d_limit) or (self.desired_z_velocity > self.z_d_limit)):
            rospy.logwarn_throttle(10.0, "z velocity setpoint outside safe region!")
            return 0.0

        # return 0.0 if z_pos is 'unsafe'
        if ((self.current_z_pos < self.deep_z_limit) or (self.current_z_pos > self.shallow_z_limit)):
            rospy.logwarn_throttle(10.0, "z outside safe region!")
            return 0.0
        
        delta_t = rospy.get_time() - self.time
        self.time = rospy.get_time()

        if self.z_uncertainty > 1.0:
            return 0
        
        if self.controller_type == 0:
            # integral-SMC
            self.e1 = self.desired_z_pos - self.current_z_pos
            self.e2 = self.desired_z_velocity - self.current_z_velocity
            self.integrator_buffer = self.sat(self.integrator_buffer+delta_t*self.sat(self.e1, self.int_sat))
            s = self.e2 + 2*self.Lambda*self.e1 + pow(self.Lambda, 2) * self.integrator_buffer
            u = self.alpha*(self.desired_z_acceleration+2*self.Lambda*self.e2+pow(self.Lambda, 2)*self.e1+self.kappa*(s/(abs(s)+self.epsilon)))

        elif self.controller_type == 1:
            # PID-Controller
            self.e1 = self.desired_z_pos - self.current_z_pos
            self.e2 = self.desired_z_velocity - self.current_z_velocity
            self.integrator_buffer = self.sat(self.integrator_buffer+delta_t*self.sat(self.e1, self.int_sat))
            u = self.k_p * self.e1 + self.k_d * self.e2 + self.k_i * self.integrator_buffer

        elif self.controller_type == 2:
            # SMC with separate i
            self.e1 = self.desired_z_pos - self.current_z_pos
            self.e2 = self.desired_z_velocity - self.current_z_velocity
            self.integrator_buffer = self.sat(self.integrator_buffer+delta_t*self.sat(self.e1, 0.05))
            s = self.e2 + self.Lambda*self.e1
            u = self.alpha*(self.desired_z_acceleration+self.Lambda*self.e2+self.kappa*(s/(abs(s)+self.epsilon))) + self.k_i * self.integrator_buffer
            
        else:
            rospy.logerr_throttle(10.0, "\nError! Undefined Controller chosen.\n")
            return 0.0
        
        k = 10**(-self.k_u*self.z_uncertainty)
        return self.sat(k*u)

    def sat(self, x, limit=1.0):
        return min(max(x, -limit), limit)


def main():
    node = ControllerNode()
    node.run()


if __name__ == "__main__":
    main()
