#!/usr/bin/env python
import rospy
import threading
from std_msgs.msg import Float64
from mavros_msgs.srv import CommandBool
from mavros_msgs.msg import MotorSetpoint
from nav_msgs.msg import Odometry
import tf
import numpy as np


class MixerNode():
    def __init__(self):
        rospy.init_node("mixer")

        self.setpoint_pub = rospy.Publisher("mavros/setpoint_motor/setpoint",
                                            MotorSetpoint,
                                            queue_size=1)

        self.simulate = rospy.get_param("simulate")

        if self.simulate:
            self.arm_vehicle()

        self.data_lock = threading.RLock()

        self.thruster = self.init_mixing()

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.thrust = 0.0
        self.vertical_thrust = 0.0
        self.lateral_thrust = 0.0

        self.roll_raw = 0.0
        self.pitch_raw = 0.0
        self.yaw_raw = 0.0
        self.thrust_raw = 0.0
        self.vertical_thrust_raw = 0.0
        self.lateral_thrust_raw = 0.0
        
        self.roll_msg_time = 0.0
        self.pitch_msg_time = 0.0
        self.yaw_msg_time = 0.0
        self.thrust_msg_time = 0.0
        self.vertical_thrust_msg_time = 0.0
        self.lateral_thrust_msg_time = 0.0

        self.state_msg_time = 0.0

        self.max_msg_timeout = 0.1

        self.estimated_state_sub = rospy.Subscriber("estimated_state",
                                                    Odometry,
                                                    self.on_state,
                                                    queue_size=1)
        self.roll_sub = rospy.Subscriber("roll",
                                         Float64,
                                         self.on_roll,
                                         queue_size=1)
        self.pitch_sub = rospy.Subscriber("pitch",
                                          Float64,
                                          self.on_pitch,
                                          queue_size=1)
        self.yaw_sub = rospy.Subscriber("yaw",
                                        Float64,
                                        self.on_yaw,
                                        queue_size=1)
        self.thrust_sub = rospy.Subscriber("thrust",
                                           Float64,
                                           self.on_thrust,
                                           queue_size=1)
        self.vertical_thrust_sub = rospy.Subscriber("vertical_thrust",
                                                    Float64,
                                                    self.on_vertical_thrust,
                                                    queue_size=1)
        self.lateral_thrust_sub = rospy.Subscriber("lateral_thrust", Float64,
                                                   self.on_lateral_thrust)

    def arm_vehicle(self): 
        # wait until the arming serivce becomes available
        rospy.wait_for_service("mavros/cmd/arming")
        # connect to the service
        arm = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        # call the service to arm the vehicle until service call was successfull
        while not arm(True).success:
            rospy.logwarn("Could not arm vehicle. Keep trying.")
            rospy.sleep(1.0)
        rospy.loginfo("Armed successfully.")

    def run(self):
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown():
            msg = self.mix()
            self.setpoint_pub.publish(msg)
            rate.sleep()

    def on_state(self, msg):
        with self.data_lock:
            self.state_msg_time = msg.header.stamp.to_sec()
            quaternion = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
            self.rot_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]

    def on_roll(self, msg):
        with self.data_lock:
            self.roll_raw = msg.data
            self.roll_msg_time = rospy.get_time()

    def on_pitch(self, msg):
        with self.data_lock:
            self.pitch_raw = msg.data
            self.pitch_msg_time = rospy.get_time()

    def on_yaw(self, msg):
        with self.data_lock:
            self.yaw_raw = msg.data
            self.yaw_msg_time = rospy.get_time()

    def on_thrust(self, msg):
        with self.data_lock:
            self.thrust_raw = msg.data
            self.thrust_msg_time = rospy.get_time()

    def on_vertical_thrust(self, msg):
        with self.data_lock:
            self.vertical_thrust_raw = msg.data
            self.vertical_thrust_msg_time = rospy.get_time()

    def on_lateral_thrust(self, msg):
        with self.data_lock:
            self.lateral_thrust_raw = msg.data
            self.lateral_thrust_msg_time = rospy.get_time()

    def transform(self):
        with self.data_lock:
            buf_lin = np.matmul(self.rot_matrix.T, self.matrix([[self.thrust_raw], [self.lateral_trust_raw], [self.vertical_thrust_raw]]))
            buf_ang = np.matmul(self.rot_matrix.T, self.matrix([[self.roll_raw], [self.pitch_raw], [self.yaw_raw]]))
            self.thrust = buf_lin[0, 0]
            self.lateral_trust = buf_lin[1, 0]
            self.vertical_thrust = buf_lin[2, 0]
            self.roll = buf_ang[0, 0]
            self.pitch = buf_ang[1, 0]
            self.yaw = buf_ang[2, 0]

    def check_msg_times(self):
        latest_time_allowed = rospy.get_time() - self.max_msg_timeout
        if self.state_msg_time < latest_time_allowed:
            rospy.logwarn_throttle(1.0, "Mixer received no Orientation Data!")
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0
            self.thrust = 0.0
            self.vertical_thrust = 0.0
            self.lateral_thrust = 0.0
            return
        if self.roll_msg_time < latest_time_allowed:
            # rospy.logwarn_throttle(1.0, "No roll control received!")
            self.roll = 0.0
        if self.pitch_msg_time < latest_time_allowed:
            # rospy.logwarn_throttle(1.0, "No pitch control received!")
            self.pitch = 0.0
        if self.yaw_msg_time < latest_time_allowed:
            rospy.logwarn_throttle(1.0, "No yaw control received!")
            self.yaw = 0.0
        if self.thrust_msg_time < latest_time_allowed:
            rospy.logwarn_throttle(1.0, "No thrust control received!")
            self.thrust = 0.0
        if self.vertical_thrust_msg_time < latest_time_allowed:
            rospy.logwarn_throttle(1.0, "No vertical_thrust control received!")
            self.vertical_thrust = 0.0
        if self.lateral_thrust_msg_time < latest_time_allowed:
            rospy.logwarn_throttle(1.0, "No lateral_thrust control received!")
            self.lateral_thrust = 0.0

    def mix(self):
        msg = MotorSetpoint()
        msg.header.stamp = rospy.Time.now()
        with self.data_lock:
            self.check_msg_times()
            for i in range(8):
                msg.setpoint[i] = 0.33 * (
                    self.roll * self.thruster[i]["roll"] +
                    self.pitch * self.thruster[i]["pitch"] +
                    self.yaw * self.thruster[i]["yaw"] +
                    self.thrust * self.thruster[i]["thrust"] +
                    self.vertical_thrust * self.thruster[i]["vertical_thrust"] +
                    self.lateral_thrust * self.thruster[i]["lateral_thrust"])
        return msg

    def init_mixing(self):
        thruster = [None] * 8
        # roll, pitch, yaw, thrust, lateral thrust, vertical thrust
        thruster[0] = dict(roll=0.0,
                           pitch=0.0,
                           yaw=1.0,
                           thrust=1.0,
                           lateral_thrust=1.0,
                           vertical_thrust=0.0)
        thruster[1] = dict(roll=0.0,
                           pitch=0.0,
                           yaw=-1.0,
                           thrust=1.0,
                           lateral_thrust=-1.0,
                           vertical_thrust=0.0)
        thruster[2] = dict(roll=0.0,
                           pitch=0.0,
                           yaw=1.0, # -1.0
                           thrust=1.0, # -1.0
                           lateral_thrust=-1.0, # 1.0
                           vertical_thrust=0.0)
        thruster[3] = dict(roll=0.0,
                           pitch=0.0,
                           yaw=-1.0, # 1.0
                           thrust=1.0, # -1.0
                           lateral_thrust=1.0, # -1.0
                           vertical_thrust=0.0)
        thruster[4] = dict(roll=-1.0,
                           pitch=-1.0,
                           yaw=0.0,
                           thrust=0.0,
                           lateral_thrust=0.0,
                           vertical_thrust=1.0)
        thruster[5] = dict(roll=-1.0,
                           pitch=1.0,
                           yaw=0.0,
                           thrust=0.0,
                           lateral_thrust=0.0,
                           vertical_thrust=-1.0)
        thruster[6] = dict(roll=1.0,
                           pitch=-1.0,
                           yaw=0.0,
                           thrust=0.0,
                           lateral_thrust=0.0,
                           vertical_thrust=-1.0)
        thruster[7] = dict(roll=1.0,
                           pitch=1.0,
                           yaw=0.0,
                           thrust=0.0,
                           lateral_thrust=0.0,
                           vertical_thrust=1.0)

        return thruster


def main():
    node = MixerNode()
    node.run()


if __name__ == "__main__":
    main()