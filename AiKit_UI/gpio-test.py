from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device
Device.pin_factory = LGPIOFactory(chip=0)

from gpiozero import LED
import time

pump_number = 71
valve_number = 72

led1 = LED(pump_number)
led2 = LED(valve_number)

try:
    while True:
        # 设置 GPIO 70 为高电平
        led1.on()
        time.sleep(0.05)
        led2.off()
        print(f"PUMP ON")
        time.sleep(5)  # 等待 1 秒

        # 设置 GPIO 70 为低电平
        led1.off()
        time.sleep(0.05)
        led2.on()
        print(f"PUMP OFF")
        time.sleep(3)  # 等待 1 秒

except KeyboardInterrupt:
    # 捕捉 Ctrl+C 并退出
    print("Exiting")

led1.close()
