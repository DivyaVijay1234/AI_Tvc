# 🚀 AI-Based Thrust Vector Control System

This project implements a real-time AI-driven thrust vector control (TVC) system for rocket stabilization using reinforcement learning (PPO) and hardware feedback via an IMU and servo-driven gimbal.

---

## 📦 Project Structure

```
AI-TVC/
├── rocket_tvc_env.py          # Custom Gym environment for TVC control
├── rocket_simulator.py        # Physics-based rocket simulation with visualization
├── train_ppo_tvc.py          # Train PPO agent in simulation
├── test_ppo_tvc.py           # Test PPO agent with real-time hardware loop
├── evaluate_tvc.py           # Evaluation loop with rendering and statistics
├── safe_training.py          # Safe training wrapper with early stopping
├── imu_reader.py             # Reads IMU pitch/yaw data over serial
├── servo_controller.py       # Sends pitch/yaw gimbal commands over serial
├── tvc_control_log.csv       # Logged control outputs
├── final_model.zip           # Final trained PPO model (for distribution)
├── best_model/               # Best model directory
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
└── README.md                # Project documentation
```

---

## 🎯 Objectives

- Build and simulate a 2-axis gimbal TVC system
- Train a PPO agent in simulation to stabilize pitch and yaw
- Replace dummy simulation values with real-time IMU feedback
- Actuate servos for gimbal pitch/yaw correction
- Log and visualize performance data
- Provide safe training with early stopping mechanisms

---

## 🎮 Simulation Features

### Physics Simulation
- Realistic rocket physics including:
  - Mass and inertia calculations
  - Thrust vector control with gimbal limits
  - Gravity and drag forces
  - Angular momentum and stability
- Configurable parameters:
  - Rocket dimensions and mass
  - Thrust magnitude (20N default)
  - Environmental conditions (gravity, air density)
  - Control limits (10° gimbal max angle)

### Real-time Visualization
- Interactive 2D visualization showing:
  - Rocket position and orientation
  - Thrust vector direction (red arrow)
  - Velocity vector (green arrow)
  - Gravity force (purple arrow)
  - Angular velocity (orange arrow)
  - Trajectory path
- Real-time performance metrics:
  - Pitch and yaw angles
  - Gimbal control inputs
  - Step counter and cumulative reward
  - Episode statistics

### Reinforcement Learning Integration
- PPO (Proximal Policy Optimization) agent for control
- Training features:
  - Custom reward function for stability optimization
  - State space: pitch, yaw, pitch_rate, yaw_rate
  - Action space: gimbal angles (pitch, yaw)
  - Soft penalties for orientation and angular velocity
  - Stability and vertical orientation bonuses
- Model management:
  - Automatic saving of best models
  - Model checkpointing during training
  - Fallback to random actions if model unavailable

---

## 🛠️ Setup Instructions

### 1. 📁 Clone and Navigate

```bash
git clone https://github.com/your-org/AI-TVC.git
cd AI-TVC
```

### 2. 🐍 Create Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. ⚙️ Run the System

#### Train PPO Agent (Simulated)
```bash
python train_ppo_tvc.py
```

#### Safe Training with Early Stopping
```bash
python safe_training.py
```

#### Test PPO Agent (With IMU + Servo Integration)
Ensure IMU sends: `pitch,yaw,pitch_rate,yaw_rate\n` via serial
Ensure servos accept: `pitch_gimbal,yaw_gimbal\n`
```bash
python test_ppo_tvc.py
```

#### Evaluate Trained Model
```bash
python evaluate_tvc.py
```

---

## 🧠 Requirements

- Python 3.8+
- Stable-Baselines3 for PPO implementation
- Matplotlib for visualization
- NumPy for numerical computations
- ESP32 (or similar) streaming IMU data via serial
- Servo controller (Arduino, PCA9685, etc.)
- USB Serial connection to host PC

---

## 📊 Model Files

The repository includes pre-trained models for immediate use:
- `final_model.zip` - Final trained PPO model
- `best_model/` - Directory containing the best performing model
- `vec_normalize.pkl` - Vector normalization parameters (excluded from git)

Training checkpoints and intermediate models are excluded from git to keep the repository clean.

---

## 🎯 Key Features

### Safety Features
- Early stopping during training to prevent overfitting
- Gimbal angle limits to prevent mechanical damage
- Episode termination on excessive orientation angles
- Maximum episode length to prevent infinite loops

### Performance Optimization
- Efficient physics simulation with configurable time steps
- Real-time visualization with trajectory tracking
- Comprehensive logging of control actions and performance
- Modular design for easy parameter tuning

### Hardware Integration
- Serial communication for IMU data input
- Servo control interface for gimbal actuation
- Real-time control loop with minimal latency
- Fallback mechanisms for hardware failures

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
