"""
gripper_block_demo.py
This module controls the robotic arm movements with adaptive waiting.

Author: Wang Weijian
Date: 2025-09-18
"""
import argparse
import time
from pymycobot import MyPalletizer260
from pymycobot.utils import get_port_list


class GripperBlockDemo:
    def __init__(self, port='COM39', baud=115200):
        self.mc = MyPalletizer260(port, baud)
        # Common positions (angles and coordinates)
        self.HOME = [0, 0, 0, -10]
        self.INIT_POSE = [0, 0, 0, -10]
        self.HOME_POSE = [0, 0, 0, -10]

        # Up and down movement
        self.UP_POINT = [165.2, 2.7, 210, -10.98]
        self.DOWN_POINT = [165.2, 2.7, 130, -10.98]

        # Point A
        self.A_TOP = [140.8, -130.4, 210, -8.08] # angles:[-42.71, 13.62, -7.99, -50.88]
        self.A_GRAB = [142.6, -131.6, 105, -8.17]  # angles:[-42.8, 32.6, 14.94, -50.88]

        # Point B
        self.B_TOP_ANGLE = [53.43, 15.73, -2.54, -40]
        self.B_PLACE = [120.5, 154.9, 110, -92.1]

    def sleep(self, t: float = 2.0):
        """Pause execution for a given number of seconds (can be float)."""
        time.sleep(t)

    def go_home(self, speed=50):
        self.mc.send_angles(self.HOME, speed)
        self.sleep()

    def move_to_init(self, speed=50):
        self.mc.send_angles(self.INIT_POSE, speed)
        self.sleep(0.5)

    def up_down_gripper(self, repeat=3, speed=90):
        """Perform up and down movement with gripper open/close"""
        for _ in range(repeat):
            self.mc.send_coords(self.DOWN_POINT, speed)
            self.sleep(1)
            self.mc.set_gripper_state(0, 80, 1)  # Close gripper

            self.mc.send_coords(self.UP_POINT, speed)
            self.sleep(1)
            self.mc.set_gripper_state(1, 80, 1)  # Open gripper

    def grab_from_A(self, speed=90):
        """Grab block from point A"""
        self.mc.send_coords(self.A_TOP, speed)
        self.sleep(1)
        self.mc.set_gripper_state(0, 80, 1)  # Open
        self.sleep(1)
        self.mc.send_coords(self.A_GRAB, speed)
        self.sleep(1)
        self.mc.set_gripper_state(1, 80, 1)  # Close
        self.sleep(1)

        self.mc.send_coords(self.A_TOP, speed)
        self.sleep(1)

    def place_to_B(self, speed=90):
        """Place block to point B"""
        self.mc.send_angles(self.B_TOP_ANGLE, speed)
        self.sleep()

        self.mc.send_coords(self.B_PLACE, speed)
        self.sleep(1)
        self.mc.set_gripper_state(0, 80, 1)  # Open
        self.sleep(1)

        self.mc.send_angles(self.B_TOP_ANGLE, 50)
        self.sleep(1)

        self.mc.send_angles(self.HOME_POSE, 50)
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

    # Initialize 260 with serial port and baud rate
    port = None
    for p in plist:
        if "ACM" in p:
            port = p
            break
    if port is None:
        raise RuntimeError("No ACM device found! Please check USB connection.")

    baud = 115200
    print(f"Using serial port: {port}, baud rate: {baud}")
    demo = GripperBlockDemo(port=port, baud=baud)
    demo.run()

