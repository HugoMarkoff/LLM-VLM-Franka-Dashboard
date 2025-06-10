# 🤖 Franka Desk Selenium Automation Suite

A robust, modular automation framework for controlling Franka Panda robots through the Franka Desk web interface using Selenium WebDriver.

## ✨ Features

- **🔧 Modular Design**: Clean separation between robot interface, commands, and automation logic
- **🎮 Interactive Mode**: Drop into a Python shell with robot command shortcuts
- **⚙️ Gripper Control**: Configure and execute gripper open/close operations with custom parameters
- **🤖 Robot Movement**: Relative motion control with configurable speed and acceleration
- **🌐 Network Management**: Automatic network configuration for robot communication
- **🔄 Persistent Sessions**: Browser session management for efficient repeated operations
- **🛡️ Robust Error Handling**: Comprehensive error handling and graceful shutdown
- **📸 Debug Support**: Automatic screenshot and HTML capture on failures

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Chrome or Chromium browser
- Network access to Franka robot (default: `172.16.0.2`)
- Sudo access for network configuration

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HugoMarkoff/Franka_selenium.git
   cd Franka_selenium
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test ChromeDriver setup:**
   ```bash
   python main.py --check-versions
   ```

## 📖 Usage

### Basic Operation Modes

#### 1. Full Automation (Legacy)
```bash
# Run complete automation sequence with GUI
python main.py

# Run headless automation
python main.py --headless

# Configure tasks only (no execution)
python main.py --config-only
```

#### 2. Initialize Robot Session
```bash
# Initialize robot and keep browser open for commands
python main.py --init-only

# Initialize with interactive Python shell
python main.py --init-only --interactive
```

#### 3. Network Configuration
```bash
# Set up network and test connectivity
python main.py --setup-network

# Skip network setup (if already configured)
python main.py --no-network-setup
```

### 🎮 Interactive Mode

The interactive mode provides a Python shell with convenient robot command shortcuts:

```bash
python main.py --init-only --interactive
```

**Available Commands:**
```python
# Gripper Configuration
config_open(speed=20)                    # Configure gripper open speed
config_close(speed=50, force=80, load=400) # Configure gripper close settings

# Gripper Execution  
open_gripper()                           # Execute gripper open
close_gripper()                          # Execute gripper close

# Robot Movement
move_robot(x=0, y=0, z=0, speed=5, accel=5) # Move robot relative position

# Status Check
robot_status()                           # Check robot status

# Full access
robot                                    # Access complete robot instance
```

**Parameter Ranges:**
- **Gripper speed**: 10-100 (speed %)
- **Gripper force**: 20-100 (grasping force in Newtons)  
- **Gripper load**: 10-1000 (load capacity in grams)
- **Robot speed**: 5-100 (movement speed %)
- **Robot acceleration**: 5-100 (acceleration %)
- **Robot x,y,z**: any float (relative movement in mm)

**Usage Examples:**
```python
# Basic gripper operations
config_open(30)                 # Set open speed to 30%
open_gripper()                  # Open gripper

config_close(60, 85, 500)       # Configure: 60% speed, 85N force, 500g load
close_gripper()                 # Close gripper

# Robot movement examples
move_robot(10, 0, 5)            # Move +10mm X, +5mm Z
move_robot(-5, 10, 0, 10, 15)   # Move -5mm X, +10mm Y, 10% speed, 15% accel
move_robot(z=-20)               # Move -20mm Z only

# Check robot status
robot_status()                  # Get current robot state

# Exit shell
exit()
```

## ⚙️ Configuration

### Network Settings

Default configuration (can be overridden with command line arguments):

```python
robot_ip = "172.16.0.2"      # Franka robot IP
local_ip = "172.16.0.1"      # Local computer IP  
interface = "enp2s0"         # Network interface
```

**Custom network settings:**
```bash
python main.py --robot-ip 192.168.1.100 --local-ip 192.168.1.10 --interface eth0
```

### Robot Authentication

Default credentials (modify in `utils/config.py`):
```python
username = "Panda"
password = "panda1234"  
```

## 🏗️ Architecture
```bash
franka-seleium/
├── main.py                 # Main entry point and argument parsing
├── utils/
│   ├── config.py          # Configuration management
│   ├── chrome_manager.py  # Chrome WebDriver management
│   ├── network_manager.py # Network configuration
│   ├── robot_interface.py # Core robot interface
│   ├── robot_commands.py  # High-level robot commands
│   ├── selenium_helper.py # Selenium utilities
│   ├── locators.py        # Web element locators
│   ├── logger.py          # Logging configuration
│   └── signal_handler.py  # Graceful shutdown handling
├── requirements.txt       # Python dependencies
└── README.md             # This file
```
