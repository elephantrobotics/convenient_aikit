"""
testtest.py
This module controls the robotic arm movements.

Author: Wang Weijian
Date: 2025-12-18
"""
# encoding: UTF-8
import platform
import cv2
import numpy as np

class DetectMarker:
    def __init__(self):
        # Camera init
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ArUco config
        self.aruco_dict = cv2.aruco.Dictionary_get(
            cv2.aruco.DICT_6X6_250
        )
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Camera intrinsics
        self.camera_matrix = np.array([
            [781.33379113, 0., 347.53500524],
            [0., 783.79074192, 246.67627253],
            [0., 0., 1.]
        ])

        self.dist_coeffs = np.array(
            [[3.41360787e-01, -2.52114260e+00,
              -1.28012469e-03, 6.70503562e-03,
              2.57018000e+00]]
        )

        # ArUco 边长（米）
        self.marker_size = 0.03

    def run(self):
        print("Start detecting ArUco markers...")

        while True:
            ret, img = self.cap.read()
            if not ret:
                print("Camera read failed")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

            if ids is not None:
                # 估计位姿
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size,
                    self.camera_matrix, self.dist_coeffs
                )

                # 画 marker
                cv2.aruco.drawDetectedMarkers(img, corners)

                for i, marker_id in enumerate(ids.flatten()):
                    tvec = tvecs[i][0]

                    # 转成 mm
                    x = round(tvec[0] * 1000, 2)
                    y = round(tvec[1] * 1000, 2)
                    z = round(tvec[2] * 1000, 2)

                    text = f"ArUco ID:{marker_id}"

                    # 打印到终端
                    print(text)

                    # 显示在图像上
                    corner = corners[i][0][0]
                    cv2.putText(
                        img,
                        text,
                        (int(corner[0]), int(corner[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("ArUco Detect", img)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    DetectMarker().run()
