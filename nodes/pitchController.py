#!/usr/bin/env python

PACKAGE = 'fav_control'
import roslib;roslib.load_manifest(PACKAGE)
import rospy

from dynamic_reconfigure.server import Server
from fav_control.cfg import PitchControlConfig

import tf

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
        self.controller_type = 1

        # PD-Controller, k_d / k_p ~= 0.6
        self.k_p = 0.1
        self.k_d = 0.03

        # SMC
        self.alpha = 0.1
        self.Lambda = 1.5
        self.kappa = 2.5
        self.epsilon = 0.4

        self.int_sat = 0.01

        self.desired_pitch = -0.5
        self.desired_pitch_vel = 0.0
        self.desired_pitch_acc = 0.0
        self.current_pitch = None
        self.current_pitch_vel = None

        self.state_msg_time = 0.0

        self.max_msg_timeout = 0.1

        self.k_i = 0.2 # 1.0
        self.integrator_buffer = 0.0

        rospy.init_node("pitchController")
        
        self.pitch_pub = rospy.Publisher("pitch",
                                        Float64,
                                        queue_size=1)
        self.error_pub = rospy.Publisher("pitch_control_error",
                                          StateVector2D,
                                          queue_size=1)
        self.controller_ready_pub = rospy.Publisher("pitch_controller_ready",
                                          Bool,
                                          queue_size=1)
        self.state_sub = rospy.Subscriber("estimated_state",
                                          Odometry,
                                          self.get_current_state,
                                          queue_size=1)

        self.time = rospy.get_time()

        rospy.sleep(5.0)
        self.report_readiness(True)

        self.server = Server(PitchControlConfig, self.server_callback)

        self.setpoint_sub = rospy.Subscriber("pitch_setpoint",
                                            StateVector3D,
                                            self.get_setpoint,
                                            queue_size=1)

    def send_control_message(self, u):
        msg = Float64()
        msg.data = u
        self.pitch_pub.publish(msg)

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
            rospy.loginfo("New Parameters received by pitch_Controller")

            # self.controller_type = config.controller_type
            
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
            self.desired_pitch = msg.position
            self.desired_pitch_vel = msg.velocity
            self.desired_pitch_acc = msg.acceleration
    
    def get_current_state(self, msg):
        with self.data_lock:
            quaternion = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
            self.current_pitch = tf.transformations.euler_from_quaternion(quaternion)[1]
            self.current_pitch_vel = msg.twist.twist.angular.y
            self.state_msg_time = rospy.get_time()

    def controller(self):
        if (rospy.get_time() - self.state_msg_time > self.max_msg_timeout):
            rospy.logwarn_throttle(1.0, "No state information received!")
            return 0.0

        if self.controller_type is None:
            rospy.logwarn_throttle(1.0, "No controller chosen!")
            return 0.0

        # return 0.0 if roll or setpoint is 'unsafe'
        if ((self.desired_pitch < self.deep_z_limit) or (self.desired_pitch > self.shallow_z_limit)):
            rospy.logwarn_throttle(1.0, "z setpoint outside safe region!")
            return 0.0

        # return 0.0 if roll or setpoint is 'unsafe'
        if ((self.current_pitch < self.deep_z_limit) or (self.current_pitch > self.shallow_z_limit)):
            rospy.logwarn_throttle(5.0, "Diving z outside safe region!")
            return 0.0
        
        delta_t = rospy.get_time() - self.time
        self.time = rospy.get_time()
        
        if self.controller_type == 0:
            # integral-SMC
            self.e1 = self.get_angular_error(self.desired_pitch, self.current_pitch)
            self.e2 = self.desired_pitch_vel - self.current_pitch_vel
            self.integrator_buffer = self.sat(self.integrator_buffer+delta_t*self.sat(self.e1, self.int_sat))
            s = self.e2 + 2*self.Lambda*self.e1 + pow(self.Lambda, 2) * self.integrator_buffer
            u = self.alpha*(self.desired_pitch_acc+2*self.Lambda*self.e2+pow(self.Lambda, 2)*self.e1+self.kappa*(s/(abs(s)+self.epsilon)))

        elif self.controller_type == 1:
            # PID-Controller
            self.e1 = self.get_angular_error(self.desired_pitch, self.current_pitch)
            self.e2 = self.desired_pitch_vel - self.current_pitch_vel
            self.integrator_buffer = self.sat(self.integrator_buffer+delta_t*self.sat(self.e1, self.int_sat))
            u = self.k_p * self.e1 + self.k_d * self.e2 + self.k_i * self.integrator_buffer

        elif self.controller_type == 2:
            # SMC with separate i
            self.e1 = self.get_angular_error(self.desired_pitch, self.current_pitch)
            self.e2 = self.desired_pitch_vel - self.current_pitch_vel
            self.integrator_buffer = self.sat(self.integrator_buffer+delta_t*self.sat(self.e1, 0.05))
            s = self.e2 + self.Lambda*self.e1
            u = self.alpha*(self.desired_pitch_acc+self.Lambda*self.e2+self.kappa*(s/(abs(s)+self.epsilon))) + self.k_i * self.integrator_buffer
            
        else:
            rospy.logerr_throttle(10.0, "\nError! Undefined Controller chosen.\n")
            return 0.0

        return self.sat(u)
    
    def get_angular_error(self, desired_angle, current_angle):
        e = (desired_angle - current_angle) % 360
        if e > 180:
            return e - 360
        elif e < -180:
            return e + 360
        else:
            return e

    def sat(self, x, limit=1.0):
        return min(max(x, -limit), limit)

def main():
   node = ControllerNode()
   node.run()


if __name__ == "__main__":
   main()