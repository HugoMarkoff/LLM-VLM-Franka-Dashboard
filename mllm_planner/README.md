# 🤖 Enhanced Robot Action Planner

A sophisticated 4-step pipeline system that combines online Vision Language Model (VLM) and Large Language Model (LLM) reasoning to generate executable robot action plans from visual scenes and task descriptions.

## ✨ Key Features

- **🧠 4-Step Pipeline Architecture**: Vision analysis + validation + reasoning + planning
- **👁️ Advanced Scene Understanding**: Qwen2.5-VL-72B for precise object detection and spatial relationships
- **🌐 Online LLM Integration**: DeepSeek Chat v3 for enhanced logical reasoning
- **🤖 Robot Execution**: Direct integration with Franka robot control interface
- **⚡ Intelligent Caching**: Vision analysis results cached for faster subsequent runs
- **📺 Flexible Output Options**: Terminal display + output capture for monitoring and logging
- **🛡️ Robust Error Handling**: Comprehensive exception handling with clear error messages
- **🔄 Multi-step Task Support**: Handles complex sequential robot tasks

## 🏗️ Architecture Overview

### 4-Step Processing Pipeline

```
📸 Image Input → 👁️ Vision Analysis → ✅ Validation → 🧠 Reasoning → 🤖 Action Plan → 🦾 Robot Execution
     ↓           (Qwen2.5-VL-72B)   (Syntax)     (DeepSeek)    (Executable)   (Franka Robot)
```

1. **STEP 1: Vision Analysis** 🔍
   - **Model**: Qwen2.5-VL-72B-Instruct (OpenRouter API)
   - **Function**: Object detection, spatial relationship analysis
   - **Output**: Structured scene description with objects and relationships

2. **STEP 2: Syntax Validation** ✅
   - **Function**: Data validation, consistency checking
   - **Process**: Object name standardization, spatial logic verification
   - **Output**: Cleaned and validated scene understanding

3. **STEP 3: Reasoning Refinement** 🧠
   - **Model**: DeepSeek Chat v3 (OpenRouter API)
   - **Function**: Logical consistency enhancement, ambiguity resolution
   - **Output**: Refined spatial understanding optimized for planning

4. **STEP 4: Action Planning** 🚀
   - **Model**: DeepSeek Chat v3 (OpenRouter API)
   - **Function**: Generate executable robot action sequences
   - **Output**: Formatted action plan with spatial constraint handling

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **OS**: Linux (tested on Ubuntu 20.04+)
- **Network**: Internet connection for API access
- **Robot**: Franka robot with web interface

### Dependencies Installation

```bash
pip install selenium requests
```

### API Configuration

You need an OpenRouter API key for model access:

```python
# In max_planner.py
OPENROUTER_API_KEY = "your-openrouter-api-key-here"
```

## 🚀 Quick Start

### Basic Usage

```python
from mllm_planner.max_planner import robot_action_planner

# Simple task
result = robot_action_planner(
    task_description="pick up the pen",
    image_path="/path/to/scene.jpg"
)

if result["success"]:
    print(f"Objects detected: {result['detected_objects']}")
    print(f"Action plan: {result['action_plan']}")
    print(f"Execution success: {result['execution_success']}")
else:
    print("Planning failed")
```

### JSON Input Usage

```python
from mllm_planner.max_planner import robot_action_planner_json

json_input = {
    "user_message": "pick up the pen",
    "imagePath": "scene.jpg",
    "objects": [
        {
            "action": "pick",
            "location": "top-center",
            "object": "pen",
            "object_center_location": {"x": 15, "y": 20, "z": 30}
        }
    ]
}

result = robot_action_planner_json(json.dumps(json_input))
```

## 🤖 Robot Actions

The system generates and executes these standardized robot actions:

| Action | Format | Description |
|--------|--------|-------------|
| `move` | `(Object, (x, y, z), move)` | Move robot to object coordinates |
| `pick` | `(Object, (x, y, z), pick)` | Pick up object at coordinates |
| `place` | `(Object, (x, y, z), place)` | Place object at coordinates |
| `OpenGripper` | `(Object, (x, y, z), OpenGripper)` | Open robot gripper |
| `CloseGripper` | `(Object, (x, y, z), CloseGripper)` | Close robot gripper |
| `Home` | `(Home, (0.0, 0.0, 0.0), Home)` | Return to home position |

### Example Action Plan
```
plan: (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 30), pick)
```

## 🧠 Planning Logic

### Spatial Constraint Handling
- **Blocking Detection**: Identifies objects on top of target objects
- **Dependency Resolution**: Automatically moves blocking objects first
- **Accessibility Analysis**: Ensures all target objects can be reached

### Multi-Step Task Processing
```python
# Single-step task
"pick up the pen" → (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 30), pick)

# Multi-step task with blocking object
"pick up the paper under the pen" → 
    (Pen, (15, 20, 30), move) -> 
    (Pen, (15, 20, 30), pick) -> 
    (Pen, (50, 50, 30), move) -> 
    (Pen, (50, 50, 30), place) -> 
    (Paper, (20, 20, 25), move) -> 
    (Paper, (20, 20, 25), pick)
```

## 📤 Return Format

```python
{
    "task_description": "pick up the pen",
    "image_path": "/path/to/image.jpg",
    "detected_objects": ["Pen", "Paper", "Controller"],
    "spatial_relationships": "The pen is on top of the paper...",
    "action_plan": "plan: (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 30), pick)",
    "success": true,
    "execution_success": true,  # Whether robot execution succeeded
    "terminal_output": "Full captured pipeline logs..."  # Only if capture_output=True
}
```

## 🛠️ Robot Integration

The planner integrates with a Franka robot through a web interface using Selenium automation:

```python
from mllm_planner.robot_action import execute_plan_from_planner

# Execute a plan on the robot
success = execute_plan_from_planner(
    plan_string="plan: (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 30), pick)",
    headless=False  # Set to True for headless browser mode
)
```

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. API Connection Issues
```python
# Error: OpenRouter API connection failed
# Solutions:
# 1. Check OPENROUTER_API_KEY in max_planner.py
# 2. Verify internet connection
# 3. Check OpenRouter API status
```

#### 2. Robot Connection Issues
```python
# Error: Failed to connect to robot
# Solutions:
# 1. Check robot IP (default: 172.16.0.2)
# 2. Verify network interface settings
# 3. Ensure robot web interface is accessible
```

#### 3. Browser Automation Issues
```python
# Error: ChromeDriver or browser issues
# Solutions:
# 1. Update Chrome and ChromeDriver
# 2. Check browser compatibility
# 3. Try headless mode: headless=True
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. Open a **Pull Request**

---

**🚀 Ready to start planning and executing robot actions? Import the module and begin with a simple task!** 