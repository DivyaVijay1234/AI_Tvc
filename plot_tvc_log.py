import matplotlib.pyplot as plt
import csv

steps, pitch, yaw, pitch_rate, yaw_rate, pitch_gimbal, yaw_gimbal = [], [], [], [], [], [], []

with open("tvc_control_log.csv", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        steps.append(int(row["Step"]))
        pitch.append(float(row["Pitch"]))
        yaw.append(float(row["Yaw"]))
        pitch_rate.append(float(row["PitchRate"]))
        yaw_rate.append(float(row["YawRate"]))
        pitch_gimbal.append(float(row["PitchGimbal"]))
        yaw_gimbal.append(float(row["YawGimbal"]))

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(steps, pitch, label="Pitch")
plt.plot(steps, yaw, label="Yaw")
plt.title("Rocket Orientation Over Time")
plt.ylabel("Angle (degrees)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(steps, pitch_gimbal, label="Pitch Gimbal")
plt.plot(steps, yaw_gimbal, label="Yaw Gimbal")
plt.title("Gimbal Commands Over Time")
plt.xlabel("Time Step")
plt.ylabel("Control Signal")
plt.legend()

plt.tight_layout()
plt.show()
