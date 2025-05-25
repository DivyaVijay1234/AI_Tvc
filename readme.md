# 🚀 AI-Based Thrust Vector Control System

This project implements a real-time AI-driven thrust vector control (TVC) system for rocket stabilization using reinforcement learning (PPO) and hardware feedback via an IMU and servo-driven gimbal.

---

## 📦 Project Structure

tvc_project/
├── rocket_tvc_env.py # Custom Gym environment with real-time IMU input
├── rocket_simulator.py # Physics-based rocket simulation
├── train_ppo_tvc.py # Train PPO agent in simulation
├── test_ppo_tvc.py # Test PPO agent with real-time hardware loop
├── evaluate_tvc.py # Simple evaluation loop with rendering
├── plot_tvc_log.py # Plot TVC control data
├── imu_reader.py # Reads IMU pitch/yaw data over serial
├── servo_controller.py # Sends pitch/yaw gimbal commands over serial
├── tvc_control_log.csv # Logged control outputs
├── ppo_tvc_model.zip # Trained PPO model
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🎯 Objectives

- Build and simulate a 2-axis gimbal TVC system
- Train a PPO agent in simulation to stabilize pitch and yaw
- Replace dummy simulation values with real-time IMU feedback
- Actuate servos for gimbal pitch/yaw correction
- Log and visualize performance data

---

## 🎮 Simulation Features

### Physics Simulation
- Realistic rocket physics including:
  - Mass and inertia calculations
  - Thrust vector control
  - Gravity and drag forces
  - Angular momentum
- Configurable parameters:
  - Rocket dimensions and mass
  - Thrust magnitude
  - Environmental conditions
  - Control limits

### Visualization
- Real-time 2D visualization of:
  - Rocket position and orientation
  - Thrust vector direction
  - Trajectory path
- Performance plots:
  - Position vs time
  - Orientation angles
  - Control inputs
  - Reward history

### Reinforcement Learning Integration
- PPO (Proximal Policy Optimization) agent for control
- Training features:
  - Custom reward function for stability
  - State space: position, velocity, orientation, angular velocity
  - Action space: gimbal angles
- Model saving and loading
- Fallback to random actions if model unavailable

---

## 🛠️ Setup Instructions

### 1. 📁 Clone and Navigate

```bash
git clone https://github.com/your-org/tvc_project.git
cd tvc_project
```

### 2. 🐍 Create Virtual Environment (optional)

```bash
python -m venv sb3env
source sb3env/bin/activate  # On Windows: sb3env\Scripts\activate
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

If you are missing pyserial, install it manually:
```bash
pip install pyserial
```

### 4. ⚙️ Run the System

#### Train PPO Agent (Simulated)
```bash
python train_ppo_tvc.py
```

#### Test PPO Agent (With IMU + Servo Integration)
Ensure IMU sends: `pitch,yaw,pitch_rate,yaw_rate\n` via serial
Ensure servos accept: `pitch_gimbal,yaw_gimbal\n`
```bash
python test_ppo_tvc.py
```

#### Run Simulation Only
```bash
python test_simulation.py
```

#### Visualize Log Output
```bash
python plot_tvc_log.py
```

---

## 🧠 Requirements

- Python 3.8+
- ESP32 (or similar) streaming IMU data via serial
- Servo controller (Arduino, PCA9685, etc.)
- USB Serial connection to host PC

---

## 📚 References

Based on research papers and IEEE surveys on deep reinforcement learning for aerospace control and real-time embedded flight systems.

---

## 👨‍💻 Authors

- Person A – Hardware & Gimbal Design
- Person B – Reinforcement Learning & Control Integration
- Guide: Dr. Sowmyarani CN

## ✅ `pyserial` Installation & Usage

### 🔧 Install

pip install pyserial
```

https://chatgpt.com/c/6830344e-e5a4-800b-a0b5-7d274698029b
