#!/usr/bin/python
# -*- coding:utf-8 -*-
# @File    : test.py
# @Author  : Wang Weijian
# @Time    :  2024/08/20 11:49:28
# @function: the script is used to do something
# @version : V1
import time

from pymycobot import MyCobot280

mc = MyCobot280('COM5', debug=False)

point_angles = [[0.61, 45.87, -92.37, -41.3, 2.02, 9.58],  # init the point
                [18.8, -7.91, -54.49, -23.02, -0.79, -14.76]]

point_coords = [[132.2, -136.9, 200.8, -178.24, -3.72, -107.17],  # D Sorting area
                [238.8, -124.1, 204.3, -169.69, -5.52, -96.52],  # C Sorting area
                [115.8, 177.3, 210.6, 178.06, -0.92, -6.11],  # A Sorting area
                [-6.9, 173.2, 201.5, 179.93, 0.63, 33.83], ]

x = 166
y = 5
mc.set_fresh_mode(0)

# print(mc.get_fresh_mode())
def check_position(data, ids):
    try:
        while True:
            res = mc.is_in_position(data, ids)
            print('res', res)
            if res == 1:
                time.sleep(0.1)
                # print(mc.get_angles())
                break
            else:
                print('notnotnot')
            time.sleep(0.1)
    except Exception as e:
        print(e)


mc.send_angles(point_angles[0], 30)
# check_position(point_angles[0], 0)
print('1')

mc.send_angles(point_angles[1], 20)
# check_position(point_angles[1], 0)
print(2)

mc.send_coords([x, y, 170.6, 179.87, -3.78, -62.75], 25, 0)
# check_position([x, y, 170.6, 179.87, -3.78, -62.75], 1)
print('3')

mc.send_coords([x, y, 100, 179.87, -3.78, -62.75], 25, 0)
check_position([x, y, 100, 179.87, -3.78, -62.75], 1)
print('4')

print('开启吸泵........')
time.sleep(2)
tmp = []
while True:
    if not tmp:
        tmp = mc.get_angles()
    else:
        break
print('tmp', tmp)
mc.send_angles([tmp[0], -0.71, -54.49, -23.02, -0.79, tmp[5]], 25)
# check_position([tmp[0], -0.71, -54.49, -23.02, -0.79, tmp[5]], 0)

mc.send_coords(point_coords[0], 40, 0)
check_position(point_coords[0], 1)

print('关闭吸泵.......')
time.sleep(0.5)

mc.send_angles(point_angles[0], 25)
