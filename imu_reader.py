import serial

def get_imu_data(port="/dev/ttyUSB0", baudrate=115200):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        line = ser.readline().decode().strip()
        pitch, yaw, pitch_rate, yaw_rate = map(float, line.split(","))
        return pitch, yaw, pitch_rate, yaw_rate
    except Exception as e:
        print(f"[IMU ERROR] {e}")
        return 0.0, 0.0, 0.0, 0.0
