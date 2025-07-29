"""
pump_demo_pi.py
This module controls the robotic arm movements.

Author: Wang Weijian
Date: 2025-07-14
"""
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)

# 开启吸泵
def pump_on():
    GPIO.output(20, 0)
    time.sleep(0.05)

# 停止吸泵
def pump_off():
    GPIO.output(20, 1)
    time.sleep(0.05)
    GPIO.output(21, 0)
    time.sleep(1)
    GPIO.output(21, 1)
    time.sleep(0.05)

for i in range(2):
    pump_on()
    time.sleep(2)
    pump_off()
    time.sleep(2)