"""
aikit_main.py
This module controls the robotic arm movements.

Author: Wang Weijian
Date: 2025-08-01
"""

import subprocess
import sys
import os
import time
from pynput import keyboard

current_process = None
device_name = None
device_key = None
in_ui_mode = False  # UI mode state
last_ui_exit_time = 0

DEVICE_MAP = {
    '1': ('AiKit_280M5', '280M5'),
    '2': ('AiKit_270M5', '270M5'),
    '3': ('AiKit_260M5', '260M5'),
}

MENU_MAP = {
    '1': """
    等待键盘输入 (按 Esc 退出)       Waiting for keyboard input (Press Esc to exit):

      1: 颜色识别                      1: Color recognition
      2: 形状识别                      2: Shape recognition
      3: AR 二维码识别                 3: AR code recognition
      4: 特征点图像识别                 4: Feature point image recognition
      5: YOLOv5 图像识别               5: YOLOv5 image recognition
      6: 启动 AiKit_UI                6: Launch AiKit_UI
      7: 启动手柄控制                  7: Launch handle control
      8: 自适应夹爪案例                 8: Adaptive gripper demo
      9: 灵巧手案例                    9: Dexterous hand demo
      0: STAG 码跟踪案例               0: STAG code tracking demo
    """,

    '2': """
    等待键盘输入 (按 Esc 退出)       Waiting for keyboard input (Press Esc to exit):

      1: 颜色识别                      1: Color recognition
      2: 形状识别                      2: Shape recognition
      3: AR 二维码识别                 3: AR code recognition
      4: 特征点图像识别                 4: Feature point image recognition
      5: YOLOv5 图像识别               5: YOLOv5 image recognition
      6: 启动 AiKit_UI                6: Launch AiKit_UI
      7: 启动手柄控制                  7: Launch handle control
      8: 自适应夹爪案例                 8: Adaptive gripper demo
      9: 灵巧手案例                    9: Dexterous hand demo
    """,

    '3': """
    等待键盘输入 (按 Esc 退出)       Waiting for keyboard input (Press Esc to exit):

      1: 颜色识别                      1: Color recognition
      2: 形状识别                      2: Shape recognition
      3: AR 二维码识别                 3: AR code recognition
      4: 特征点图像识别                 4: Feature point image recognition
      5: YOLOv5 图像识别               5: YOLOv5 image recognition
      6: 启动 AiKit_UI                6: Launch AiKit_UI
      7: 启动手柄控制                  7: Launch handle control
      8: 自适应夹爪案例                 8: Adaptive gripper demo
    """,
}

# Script path splicing function
def get_script_path(script_name):
    if not device_name:
        print("请先选择设备！Please select a device first!")
        return None
    return os.path.join('/home/er/convenient_aikit', device_name, 'scripts', script_name)

# Start script function
def run_script(script_path, use_sudo=False):
    global current_process, in_ui_mode, last_ui_exit_time

    if current_process is not None and current_process.poll() is None:
        print("终止当前算法进程... Terminating current algorithm process...")
        current_process.terminate()
        current_process.wait()

    if not script_path:
        return

    print(f"启动脚本: {script_path} Starting script: {script_path} ")
    current_python = sys.executable
    if use_sudo:
        current_process = subprocess.Popen(['sudo', current_python, script_path])
    else:
        current_process = subprocess.Popen([current_python, script_path])

# Key response
def on_press(key):
    global current_process, in_ui_mode, last_ui_exit_time

    try:
        # Ignore all key presses for 0.5 seconds after the UI exits
        if time.time() - last_ui_exit_time < 0.5:
            # print("Ignore the keystroke residue at the moment of UI exit")
            return

        if hasattr(key, 'char'):
            # Disable algorithm switching in UI mode
            if in_ui_mode and key.char in ['1', '2', '3', '4', '5', '7', '8', '9', '0']:
                # print("Currently in UI mode, ignoring numeric key input")
                return
            if key.char == '1':
                run_script(get_script_path('aikit_color.py'), use_sudo=False)
            elif key.char == '2':
                run_script(get_script_path('aikit_shape.py'), use_sudo=False)
            elif key.char == '3':
                run_script(get_script_path('aikit_encode.py'), use_sudo=False)
            elif key.char == '4':
                run_script(get_script_path('aikit_img.py'), use_sudo=False)
            elif key.char == '5':
                run_script(get_script_path('yolov5_img.py'), use_sudo=False)
            elif key.char == '6':
                in_ui_mode = True
                run_script('/home/er/convenient_aikit/AiKit_UI/main.py', use_sudo=False)
                if current_process:
                    current_process.wait()
                in_ui_mode = False
                last_ui_exit_time = time.time()  #Record UI exit time
                print("UI 模式结束，恢复数字键切换功能 UI mode ended, numeric key switching re-enabled")
            elif key.char == '7':
                handle_path = os.path.join('/home/er/convenient_aikit/handle_control', f'{device_key}_wireless_keyboard_mouse_handle_control_raspi_linux.py')
                run_script(handle_path, use_sudo=False)
            elif key.char == '8':
                run_script(get_script_path('gripper_block_demo.py'), use_sudo=False)
            elif key.char == '9':
                run_script(get_script_path('dance_action_finger_demo.py'), use_sudo=False)
            elif key.char == '0':
                run_script(get_script_path('camera_detect.py'), use_sudo=False)
            else:
                print(f"无效按键：{key.char}，请按 0-9 或 Esc Invalid key: {key.char}, please press 0-9 or Esc")

        elif key == keyboard.Key.esc:
            print("退出监听 Exiting listener")
            if current_process is not None and current_process.poll() is None:
                print("终止当前算法脚本... Terminating current algorithm script...")
                current_process.terminate()
                current_process.wait()
            return False
        else:
            print(f"忽略特殊按键：{key} Ignoring special key: {key}")

    except Exception as e:
        print(f"按键监听出错: {e} Key listener error: {e}")
        return False


if __name__ == '__main__':
    while True:
        print("请选择设备：Please select a device:")
        print("1 - myCobot 280 M5")
        print("2 - MechArm 270 M5")
        print("3 - MyPalletizer 260 M5")
        print("q - 退出程序 q - Quit the program" )
        device_input = input("输入设备编号（1/2/3 或 q 退出）：Enter device number (1/2/3 or q to quit):").strip()

        if device_input == 'q':
            print("已退出程序。Program exited.")
            sys.exit(0)
        if device_input in DEVICE_MAP:
            break
        else:
            print("无效的设备编号，请重新输入！Invalid device number, please try again!\n")

    device_name, device_key = DEVICE_MAP[device_input]
    print(f"当前选择设备: {device_key} Current selected device: {device_key}")

    print(MENU_MAP[device_input])

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()