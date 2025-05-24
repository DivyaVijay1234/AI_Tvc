import serial

def send_to_servos(pitch, yaw, port="/dev/ttyUSB0", baudrate=115200):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        command = f"{pitch:.2f},{yaw:.2f}\n"
        ser.write(command.encode())
    except Exception as e:
        print(f"[SERVO ERROR] {e}")
