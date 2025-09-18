"""
dance_action_finger_demo.py
This module demonstrates the movement of a five-finger dexterous robotic hand.

Author: Wang Weijian
Date: 2025-09-18
"""
import argparse
import time
from pymycobot import MechArm270
from pymycobot.utils import get_port_list

# List available serial ports
plist = get_port_list()
print('serial:', plist)

port = None
for p in plist:
    if "ACM" in p:
        port = p
        break
if port is None:
    raise RuntimeError("No ACM device found! Please check USB connection.")

baud = 115200
print(f"Using serial port: {port}, baud rate: {baud}")

# Initialize the MyCobot280 robotic arm
mc = MechArm270(port, baud)

# Ensure the fresh mode is set to 0
if mc.get_fresh_mode() != 0:
    mc.set_fresh_mode(0)

# Move the robotic arm to the initial position
mc.send_angles([0, 0, 0, 0, 0, -40], 50)
time.sleep(2)

# Move the robotic arm to a specific position for gripping
mc.send_angles([-90, 0, 0, 0, 0, -40], 80)
mc.set_gripper_state(1, 80, 2)  # Close gripper with specified speed and duration
time.sleep(2)

# Repeat a demonstration sequence three times
for i in range(3):
    # Move quickly to the first demonstration position
    mc.send_angles([-89.64, -50.27, 49.83, -1.84, -0.7, -40], 90)
    mc.set_gripper_state(1, 80, 2)
    time.sleep(0.5)

    # Move quickly to the second demonstration position
    mc.send_angles([-90.0, 59.85, -65.03, -3.6, 7.2, -39.72], 90)
    mc.set_gripper_state(1, 80, 2)
    time.sleep(0.5)

# Pause and then reset the robotic arm to the initial position
time.sleep(2)
mc.send_angles([0, 0, 0, 0, 0, -40], 50)
mc.set_gripper_state(0, 80, 2)  # Open gripper
time.sleep(2)