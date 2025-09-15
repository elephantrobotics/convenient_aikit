"""
280PI_aikit_main.py
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
in_ui_mode = False  # UI mode state
last_ui_exit_time = 0


BASE_DIR = "/home/er/convenient_aikit/AiKit_280PI/scripts"
HANDLE_DIR = "/home/er/convenient_aikit/handle_control"
UI_PATH = "/home/er/convenient_aikit/AiKit_UI/main.py"
DEVICE_KEY = "280PI"

# Script path splicing function
def run_script(script_path, use_sudo=False):
    global current_process, in_ui_mode, last_ui_exit_time

    if current_process is not None and current_process.poll() is None:
        print("终止当前算法进程...")
        current_process.terminate()
        current_process.wait()

    if not script_path:
        return

    print(f"启动脚本: {script_path}")
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
                run_script(os.path.join(BASE_DIR, 'aikit_color.py'))
            elif key.char == '2':
                run_script(os.path.join(BASE_DIR, 'aikit_shape.py'))
            elif key.char == '3':
                run_script(os.path.join(BASE_DIR, 'aikit_encode.py'))
            elif key.char == '4':
                run_script(os.path.join(BASE_DIR, 'aikit_img.py'))
            elif key.char == '5':
                run_script(os.path.join(BASE_DIR, 'yolov5_img.py'))
            elif key.char == '6':
                in_ui_mode = True
                run_script(UI_PATH)
                if current_process:
                    current_process.wait()
                in_ui_mode = False
                last_ui_exit_time = time.time()  # Record UI exit time
                print("UI 模式结束，恢复数字键切换功能")
            elif key.char == '7':
                handle_script = os.path.join(HANDLE_DIR, f"{DEVICE_KEY}_wireless_keyboard_mouse_handle_control_raspi_linux.py")
                run_script(handle_script)
            elif key.char == '8':
                run_script(os.path.join(BASE_DIR, 'gripper_block_demo.py'))
            elif key.char == '9':
                run_script(os.path.join(BASE_DIR, 'dance_action_finger_demo.py'))
            elif key.char == '0':
                run_script(os.path.join(BASE_DIR, 'camera_detect.py'))
            else:
                print(f"无效按键：{key.char}，请按 0-9 或 Esc")

        elif key == keyboard.Key.esc:
            print("退出监听")
            if current_process is not None and current_process.poll() is None:
                print("终止当前算法脚本...")
                current_process.terminate()
                current_process.wait()
            return False
        else:
            print(f"忽略特殊按键：{key}")

    except Exception as e:
        print(f"按键监听出错: {e}")
        return False


if __name__ == '__main__':
    menu = """
    等待键盘输入 (按 Esc 退出):

      1: 颜色识别
      2: 形状识别
      3: AR二维码识别
      4: 特征点图像识别
      5: YOLOv5 图像识别
      6: 启动 AiKit_UI
      7: 启动手柄控制
      8: 自适应夹爪案例
      9: 灵巧手案例
      0: STAG 码跟踪案例
    """
    print(menu)
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
