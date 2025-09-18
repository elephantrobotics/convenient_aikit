"""
gripper_block_demo.py
This module controls the robotic arm movements with adaptive waiting.

Author: Wang Weijian
Date: 2025-09-18
"""
import argparse
import time
from pymycobot import MechArm270
from pymycobot.utils import get_port_list


class GripperBlockDemo:
    def __init__(self, port='COM5', baud=115200):
        self.mc = MechArm270(port, baud)
        if self.mc.get_fresh_mode() != 0:
            self.mc.set_fresh_mode(0)
        # Common positions (angles and coordinates)
        self.HOME = [0, 0, 0, 0, 0, 0]
        self.INIT_POSE = [0, 0, 0, 0, 80, 0]
        self.HOME_POSE = [0, 0, 0, 0, 80, 0]

        # Up and down movement
        self.UP_POINT = [118.1, 0.6, 210, -179.76, 7.55, -179.78]
        self.DOWN_POINT = [118.1, 0.6, 130, -179.76, 7.55, -179.78]

        # Point A
        self.A_TOP = [130.4, -125.0, 161, -173.78, 5.84, -176.86] # [-43.76, 42.01, -40.6, 0.08, 82.08, -46.49]
        self.A_GRAB = [130.4, -125.0, 130, -173.78, 5.84, -176.86] # [-43.76, 42.8, -21.7, 0.08, 62.57, -46.49]

        # Point B
        self.B_TOP_ANGLE = [55.28, 50.71, -56.95, -0.79, 86.92, -35.68]
        self.B_PLACE = [110.7, 158.2, 130, -173.89, 7.09, -88.67]

    def sleep(self, t: float = 2.0):
        """Pause execution for a given number of seconds (can be float)."""
        time.sleep(t)

    def go_home(self, speed=50):
        self.mc.send_angles(self.HOME_POSE, speed)
        self.sleep()

    def move_to_init(self, speed=50):
        self.mc.send_angles(self.INIT_POSE, speed)
        self.sleep(0.5)

    def up_down_gripper(self, repeat=3, speed=90):
        """Perform up and down movement with gripper open/close"""
        for _ in range(repeat):
            self.mc.send_coords(self.DOWN_POINT, speed, 1)
            self.sleep(1)
            self.mc.set_gripper_state(0, 80, 1)  # Open gripper

            self.mc.send_coords(self.UP_POINT, speed, 1)
            self.sleep(1)
            self.mc.set_gripper_state(1, 80, 1)  # Close gripper

    def grab_from_A(self, speed=90):
        """Grab block from point A"""
        self.mc.send_coords(self.A_TOP, speed, 1)
        self.sleep(1.5)
        self.mc.set_gripper_state(0, 80, 1)  # Open
        self.sleep(1)
        self.mc.send_coords(self.A_GRAB, speed, 1)
        self.sleep(1)
        self.mc.set_gripper_state(1, 80, 1)  # Close
        self.sleep(1)

        self.mc.send_coords(self.A_TOP, speed, 1)
        self.sleep(1)

    def place_to_B(self, speed=90):
        """Place block to point B"""
        self.mc.send_angles(self.B_TOP_ANGLE, 50)
        self.sleep()

        self.mc.send_coords(self.B_PLACE, speed, 1)
        self.sleep(1)
        self.mc.set_gripper_state(0, 80, 1)  # Open
        self.sleep(1)

        self.mc.send_angles(self.B_TOP_ANGLE, 50)
        self.sleep(1)

        self.mc.send_angles(self.HOME_POSE, speed)
        self.sleep()

    def run(self):
        """Execute full demo process"""
        print(">>> Go home")
        self.go_home(90)

        print(">>> Move to init pose")
        self.move_to_init(50)

        print(">>> Up-down movement with gripper actions")
        self.up_down_gripper(3, 90)

        print(">>> Grab block from A")
        self.grab_from_A(90)

        print(">>> Place block to B")
        self.place_to_B(90)

        print(">>> Back to home")
        self.go_home(50)

if __name__ == "__main__":
    # List available serial ports
    plist = get_port_list()
    print('serial:', plist)

    # Initialize 270 with serial port and baud rate
    port = '/dev/ttyAMA0'
    baud = 1000000
    print(f"Using serial port: {port}, baud rate: {baud}")
    demo = GripperBlockDemo(port=port, baud=baud)
    demo.run()

