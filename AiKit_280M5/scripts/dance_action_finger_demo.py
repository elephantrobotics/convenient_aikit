"""
dance_action_finger_demo.py
This module demonstrates the movement of a five-finger dexterous robotic hand.

Author: Wang Weijian
Date: 2025-09-03
"""
import argparse
import time
from pymycobot import MyCobot280
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
mc = MyCobot280(port, baud)

# Ensure the fresh mode is set to 0
if mc.get_fresh_mode() != 0:
    mc.set_fresh_mode(0)

# Move the robotic arm to the initial position
mc.send_angles([0, 0, 0, 0, 0, 0], 50)
time.sleep(2)

# Move to a demonstration pose
mc.send_angles([-90, 0, 0, 90, 0, 0], 80)
time.sleep(1.5)

# Perform a simple finger movement sequence three times
for i in range(3):
    mc.send_angles([-30, 0, 0, 90, -30, 0], 100)
    time.sleep(1)
    mc.send_angles([-120, 0, 0, 90, 30, 0], 100)
    time.sleep(1)

# Move the robotic arm to a specific position for gripping
mc.send_angles([-90, 104.41, -129.46, 30.14, 0, 0], 80)
mc.set_gripper_state(1, 80, 2)  # Close gripper with specified speed and duration
time.sleep(2)

# Repeat a demonstration sequence three times
for i in range(3):
    # Move quickly to the first demonstration position
    mc.send_angles([-90, 104.41, -129.46, 30.14, 0, 0], 90)
    mc.set_gripper_state(1, 80, 2)
    time.sleep(0.5)

    # Change LED color to blue
    mc.set_color(0, 0, 50)
    time.sleep(0.2)

    # Move quickly to the second demonstration position
    mc.send_angles([-90, -30.41, -70.92, 102.39, 0, 0], 90)
    mc.set_gripper_state(1, 80, 2)
    time.sleep(0.5)

    # Change LED color to green
    mc.set_color(0, 50, 0)
    time.sleep(0.2)

# Pause and then reset the robotic arm to the initial position
time.sleep(2)
mc.send_angles([0, 0, 0, 0, 0, 0], 50)
mc.set_gripper_state(0, 80, 2)  # Open gripper
time.sleep(2)