"""
pump_demo_m5.py
This module controls the robotic arm movements.

Author: Wang Weijian
Date: 2025-07-14
"""
from pymycobot import MyCobot280,utils, MyPalletizer260
import time

arm = MyPalletizer260(utils.get_port_list()[0])

# 开启吸泵
def pump_on():
    arm.set_basic_output(5, 0)
    time.sleep(0.05)

# 停止吸泵
def pump_off():
    arm.set_basic_output(5, 1)
    time.sleep(0.05)
    arm.set_basic_output(2, 0)
    time.sleep(1)
    arm.set_basic_output(2, 1)
    time.sleep(0.05)

for i in range(2):
    pump_on()
    time.sleep(2)
    pump_off()
    time.sleep(2)