#!/usr/bin/env python

PACKAGE = 'fav_control'
import roslib;roslib.load_manifest(PACKAGE)
import rospy
import numpy as np

import threading

from dynamic_reconfigure.server import Server
from fav_control.cfg import SetpointConfig

from fav_control.msg import StateVector3D

class SetpointsNode():
    def __init__(self):
        self.data_lock = threading.RLock()

        rospy.init_node("setpoints")
        self.x_setpoint_pub = rospy.Publisher("x_setpoint", StateVector3D, queue_size=1)
        self.y_setpoint_pub = rospy.Publisher("y_setpoint", StateVector3D, queue_size=1)
        self.z_setpoint_pub = rospy.Publisher("z_setpoint", StateVector3D, queue_size=1)
        self.roll_setpoint_pub = rospy.Publisher("roll_setpoint", StateVector3D, queue_size=1)
        self.pitch_setpoint_pub = rospy.Publisher("pitch_setpoint", StateVector3D, queue_size=1)
        self.yaw_setpoint_pub = rospy.Publisher("yaw_setpoint", StateVector3D, queue_size=1)

        self.x_setpoint_default = np.array([0.7, 0.0, 0.0])
        self.y_setpoint_default = np.array([2.0, 0.0, 0.0])
        self.z_setpoint_default = np.array([-0.6, 0.0, 0.0])
        self.roll_setpoint_default = np.array([0, 0.0, 0.0])
        self.pitch_setpoint_default = np.array([0, 0.0, 0.0])
        self.yaw_setpoint_default = np.array([np.pi/2, 0.0, 0.0])

        self.x_setpoint = self.x_setpoint_default.copy()
        self.y_setpoint = self.y_setpoint_default.copy()
        self.z_setpoint = self.z_setpoint_default.copy()
        self.roll_setpoint = self.roll_setpoint_default.copy()
        self.pitch_setpoint = self.pitch_setpoint_default.copy()
        self.yaw_setpoint = self.yaw_setpoint_default.copy()

        self.x_trajectory = 0
        self.x_mean = None
        self.x_amplitude = None
        self.x_omega = None
        self.x_period_time = None
        self.x_wave_init_time = None
        
        self.y_trajectory = 0
        self.y_mean = None
        self.y_amplitude = None
        self.y_omega = None
        self.y_period_time = None
        self.y_wave_init_time = None

        self.z_trajectory = 0
        self.z_mean = None
        self.z_amplitude = None
        self.z_omega = None
        self.z_period_time = None
        self.z_wave_init_time = None
        
        self.yaw_sine_wave = False
        self.yaw_mean = None
        self.yaw_amplitude = None
        self.yaw_omega = None
        self.yaw_wave_init_time = None

        self.server = Server(SetpointConfig, self.server_callback)

    def run(self):
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            with self.data_lock:
                time = rospy.get_time()

                if self.x_trajectory == 1:
                    self.x_setpoint = self.sine_wave(self.x_mean, self.x_amplitude, self.x_omega, time-self.x_wave_init_time)
                elif self.x_trajectory == 2:
                    self.x_setpoint = self.step_wave(self.x_mean, self.x_amplitude, self.x_period_time, time-self.x_wave_init_time)
                x_msg = StateVector3D()
                x_msg.header.stamp = rospy.Time.now()
                x_msg.position = self.x_setpoint[0]
                x_msg.velocity = self.x_setpoint[1]
                x_msg.acceleration = self.x_setpoint[2]
                self.x_setpoint_pub.publish(x_msg)

                if self.y_trajectory == 1:
                    self.y_setpoint = self.sine_wave(self.y_mean, self.y_amplitude, self.y_omega, time-self.y_wave_init_time)
                elif self.y_trajectory == 2:
                    self.y_setpoint = self.step_wave(self.y_mean, self.y_amplitude, self.y_period_time, time-self.y_wave_init_time)
                y_msg = StateVector3D()
                y_msg.header.stamp = rospy.Time.now()
                y_msg.position = self.y_setpoint[0]
                y_msg.velocity = self.y_setpoint[1]
                y_msg.acceleration = self.y_setpoint[2]
                self.y_setpoint_pub.publish(y_msg)

                if self.z_trajectory == 1:
                    self.z_setpoint = self.sine_wave(self.z_mean, self.z_amplitude, self.z_omega, time-self.z_wave_init_time)
                elif self.z_trajectory == 2:
                    self.z_setpoint = self.step_wave(self.z_mean, self.z_amplitude, self.z_period_time, time-self.z_wave_init_time)
                z_msg = StateVector3D()
                z_msg.header.stamp = rospy.Time.now()
                z_msg.position = self.z_setpoint[0]
                z_msg.velocity = self.z_setpoint[1]
                z_msg.acceleration = self.z_setpoint[2]
                self.z_setpoint_pub.publish(z_msg)

                roll_msg = StateVector3D()
                roll_msg.header.stamp = rospy.Time.now()
                roll_msg.position = self.roll_setpoint[0]
                roll_msg.velocity = self.roll_setpoint[1]
                roll_msg.acceleration = self.roll_setpoint[2]
                self.roll_setpoint_pub.publish(roll_msg)

                pitch_msg = StateVector3D()
                pitch_msg.header.stamp = rospy.Time.now()
                pitch_msg.position = self.pitch_setpoint[0]
                pitch_msg.velocity = self.pitch_setpoint[1]
                pitch_msg.acceleration = self.pitch_setpoint[2]
                self.pitch_setpoint_pub.publish(pitch_msg)

                if self.yaw_sine_wave:
                    self.yaw_setpoint = self.sine_wave(self.yaw_mean, self.yaw_amplitude, self.yaw_omega, time-self.yaw_wave_init_time)
                yaw_msg = StateVector3D()
                yaw_msg.header.stamp = rospy.Time.now()
                yaw_msg.position = self.yaw_setpoint[0]
                yaw_msg.velocity = self.yaw_setpoint[1]
                yaw_msg.acceleration = self.yaw_setpoint[2]
                self.yaw_setpoint_pub.publish(yaw_msg)

            rate.sleep()

    def server_callback(self, config, level):
        with self.data_lock:
            rospy.loginfo("New Parameters received by Setpoint Generator")

            time = rospy.get_time()

            self.x_trajectory = config.x_trajectory
            if self.x_trajectory == 0:
                self.x_setpoint[0] = config.x_setpoint
                self.x_wave_init_time = None
            else:
                if self.x_wave_init_time is None:
                    self.x_wave_init_time = time
                self.x_mean = config.x_mean
                self.x_amplitude = config.x_amplitude
                x_frequency = config.x_frequency
                self.x_omega = 2*np.pi*x_frequency
                self.x_period_time = 1.0/x_frequency
            
            self.y_trajectory = config.y_trajectory
            if self.y_trajectory == 0:
                self.y_setpoint[0] = config.y_setpoint
                self.y_wave_init_time = None
            else:
                if self.y_wave_init_time is None:
                    self.y_wave_init_time = time
                self.y_mean = config.y_mean
                self.y_amplitude = config.y_amplitude
                y_frequency = config.y_frequency
                self.y_omega = 2*np.pi*y_frequency
                self.y_period_time = 1.0/y_frequency
            
            self.z_trajectory = config.z_trajectory
            if self.z_trajectory == 0:
                self.z_setpoint[0] = config.z_setpoint
                self.z_wave_init_time = None
            else:
                if self.z_wave_init_time is None:
                    self.z_wave_init_time = time
                self.z_mean = config.z_mean
                self.z_amplitude = config.z_amplitude
                z_frequency = config.z_frequency
                self.z_omega = 2*np.pi*z_frequency
                self.z_period_time = 1.0/z_frequency
            
            self.yaw_sine_wave = config.yaw_sine_wave
            if not self.yaw_sine_wave:
                self.yaw_setpoint[0] = (np.pi/180) * config.yaw_setpoint
                self.yaw_wave_init_time = None
            else:
                if self.yaw_wave_init_time is None:
                    self.yaw_wave_init_time = time
                self.yaw_mean = np.pi/2
                self.yaw_amplitude = (np.pi/180) * config.yaw_amplitude
                self.yaw_omega = 2*np.pi*config.yaw_frequency

        return config

    def sine_wave(self, mean, amplitude, omega, t):
        pos = mean + amplitude * np.sin(omega*t)
        vel = omega*amplitude * np.cos(omega*t)
        acc = -pow(omega, 2)*amplitude * np.sin(omega*t)
        return np.array([pos, vel, acc])
    
    def step_wave(self, mean, amplitude, period_time, t):
        t_star = t % period_time
        if t_star <= (period_time/2):
            pos = mean + amplitude
        else:
            pos = mean - amplitude
        return np.array([pos, 0, 0])


def main():
   node = SetpointsNode()
   node.run()


if __name__ == "__main__":
   main()
