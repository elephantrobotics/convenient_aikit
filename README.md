# **AiKit UI Introduction**

Use Python + OpenCV + PyQt5 to perform color recognition, shape recognition, feature point image recognition, AR QR code recognition, and YOLOv5 image recognition on a robotic arm.

- **Applicable Devices:**
  - myPalletizer 260 for M5
  - myCobot 280 for M5
  - mechArm 270 for M5
  - myCobot 280 for Pi
  - mechArm 270 for Pi
  - myPalletizer 260 for Pi

## 1 Operating Environment

Linux Raspberry Pi, Windows 10 or Windows 11. The end effector only supports the myCobot Vertical Suction Pump 2.0.

## 2 Installing the Code

```bash
git clone -b Convenient_AiKit https://github.com/elephantrobotics/AiKit_UI.git
```

## 3 Python Installing Dependencies

```bash
cd AiKit_UI
pip install -r requirements.txt
```

## 4 Launching the Program

Switch to the project folder and run the main.py script

```bash
cd AiKit_UI
python main.py
```

After the startup is successful, as shown in the figure below:<br>

![img](./libraries/AiKit_UI_img/EN/1.png) 

### **Features**

#### **language switch**

Click the button in the upper right corner of the window to switch between languages (Chinese, English).<br>
![img](./libraries/AiKit_UI_img/EN/27.png)

#### **device connection**

1. Select serial port, device, baud rate<br>![img](./libraries/AiKit_UI_img/EN/2.png)
2. Click the 'CONNECT' button to connect, after the connection is successful, the 'CONNECT' button will change to 'DISCONNECT'<br>
   ![img](./libraries/AiKit_UI_img/EN/3.png)

3. Clicking the 'DISCONNECT' button will disconnect the robot arm<br>
   ![img](./libraries/AiKit_UI_img/EN/4.png)

4. After the robotic arm is successfully connected, the gray button will be lit and become clickable.<br>
   ![img](./libraries/AiKit_UI_img/EN/5.png)

#### **Turn on the camera**

1. Set the camera serial number, the default serial number is 0, when Windows is used, the serial number is usually 1, and when Linux is used, the serial number is usually 0.<br>
   ![img](./libraries/AiKit_UI_img/EN/6.png)

2. Click the 'Open' button to try to open the camera. If the opening fails, you should try to change the camera serial number; the camera is successfully opened as shown in the figure below: Note: Before use, the camera should be adjusted to be just above the QR code whiteboard, and there is a line The straight line is facing the mechanical arm.<br>
   ![img](./libraries/AiKit_UI_img/EN/7.png)

3. After successfully opening the camera, click the 'Close' button to close the camera<br>
   ![img](./libraries/AiKit_UI_img/EN/8.png)

#### **algorithm control**

1. Fully automatic mode, after clicking the 'Auto Mode' button, the recognition, grabbing, and placing will always be on; click the 'Auto Mode' button again to turn off the fully automatic mode.<br>
   ![img](./libraries/AiKit_UI_img/EN/9.png)

2. Go back to the initial point of grabbing, click the 'Go' button, it will stop the current operation and return to the initial point.<br>![img](./libraries/AiKit_UI_img/EN/10.png)

3. Step-by-step 
   Recognition recognition: click the 'Run' button to start the recognition, Aigorithm is the current algorithm used. <br>
   ![img](./libraries/AiKit_UI_img/EN/11.png)
   Pick: Click the 'Run' button to start the capture. After the capture is successful, the recognition and capture will be automatically closed, and you need to click it again for the next use. <br>
   ![img](./libraries/AiKit_UI_img/EN/12.png)
   Placement: Click the 'Run' button to start placing. The BinA, BinB, BinC, and BinD selection boxes correspond to BinA, BinB, BinC, and BinD 4 storage boxes, respectively, and will be placed in the designated storage box after selection.<br>
   ![img](./libraries/AiKit_UI_img/EN/13.png)

4. Grasping point adjustment, X offset, Y offset, Z offset respectively represent the position of the X-axis, Y-axis, and Z-axis of the robot arm coordinates. They can be modified according to actual needs. Click the 'Save' button Save it. After successful saving, it will be captured according to the latest point.<br>
   ![img](./libraries/AiKit_UI_img/EN/14.png)<br>
   ![img](./libraries/AiKit_UI_img/EN/15.png)

5. Open the file location, our code is open source, you can modify it according to your needs, click the 'Open File' button to open the file location.<br> ![img](./libraries/AiKit_UI_img/EN/16.png)
   Open the 'main.py' file and modify it <br>
   ![img](./libraries/AiKit_UI_img/EN/17.png)<br>

6. Algorithm selection includes color recognition, shape recognition, two-dimensional code recognition, Keypoints recognition and yolov5 recognition. Selecting the corresponding algorithm will perform corresponding recognition.<br>
   ![img](./libraries/AiKit_UI_img/EN/18.png)

7. Add a picture for 'Keypoints' <br>
   ![img](./libraries/AiKit_UI_img/EN/19.png)
   Click the 'Add' button, the camera will open and a prompt will appear. <br>
   ![img](./libraries/AiKit_UI_img/EN/20.png)
   Click the 'Cut' button, the current camera content will be intercepted, and a prompt will be given to 'press the ENTER key after the content needs to be saved'<br>

   ![img](./libraries/AiKit_UI_img/EN/21.png)
   Frame the content to be saved and press the ENTER key to start selecting the saved area, corresponding to BinA, BinB, BinC, BinD 4 storage boxes.<br>

   ![img](./libraries/AiKit_UI_img/EN/22.png)
   The intercepted content will be displayed here<br>
   ![img](./libraries/AiKit_UI_img/EN/23.png)

   You can enter the following path to view the saved pictures<br>
   ![img](./libraries/AiKit_UI_img/EN/24.png)

8. Click the 'Exit' button to exit adding pictures. Note: If you start capturing, please exit after capturing. You can choose not to save the captured pictures.<br>
   ![img](./libraries/AiKit_UI_img/EN/19.png)

#### **coordinate display**

1. Real-time coordinate display of the robotic arm: click the 'current coordinates' button to open<br>![img](./libraries/AiKit_UI_img/EN/25.png)

2. Recognition coordinate display: click the ''image coordinates' button to open<br>
   ![image-20230106180304086](./libraries/AiKit_UI_img/EN/26.png)