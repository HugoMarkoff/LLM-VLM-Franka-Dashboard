#!/usr/bin/env python3
"""
Demo: Simplified Integration between max_planner.py and robot_action.py

This demonstrates how max_planner.py now automatically calls robot_action.py
to execute the generated plan without needing any flags or parameters.
"""

import json

# Example JSON input for the planner
example_task = {
    "user_message": "pick up the pen",
    "imagePath": "old.jpeg",
    "objects": [
        {
            "action": "pick",
            "location": "top-center",
            "object": "pen",
            "object_center_location": {"x": 15, "y": 20, "z": 30},
            "object_size": {"width": 5, "length": 15},
            "object_orientation": 0
        },
        {
            "action": "pick",
            "location": "bottom-center", 
            "object": "paper",
            "object_center_location": {"x": 25, "y": 25, "z": 25},
            "object_size": {"width": 20, "length": 30},
            "object_orientation": 0
        }
    ]
}

print("🚀 SIMPLIFIED INTEGRATION DEMO")
print("=" * 50)
print("This shows how max_planner.py automatically calls robot_action.py")
print("after generating a plan, without needing any flags or parameters.")
print()

print("📋 Example Task:")
print(f"   Task: {example_task['user_message']}")
print(f"   Image: {example_task['imagePath']}")
print(f"   Objects: {len(example_task['objects'])} objects with coordinates")
print()

print("🔧 How it works:")
print("   1. max_planner.py generates a plan from the image and task")
print("   2. At the end of planning, it automatically calls:")
print("      robot_action.execute_plan_from_planner(result['action_plan'])")
print("   3. robot_action.py executes the plan on the Franka robot")
print("   4. Returns success/failure status")
print()

print("💻 Command line usage:")
json_string = json.dumps(example_task)
print(f"   python mllm_planner/max_planner.py '{json_string}'")
print()

print("📝 Code Flow:")
print("   max_planner.py:")
print("   ├── Generate plan using VLM + LLM")
print("   ├── if final_plan:")
print("   │   ├── robot_action.execute_plan_from_planner(final_plan)")
print("   │   └── result['execution_success'] = success")
print("   └── return result")
print()
print("   robot_action.py:")
print("   ├── execute_plan_from_planner(plan_string):")
print("   ├── ├── Initialize robot automation")
print("   ├── ├── Parse plan string")
print("   ├── ├── Execute actions sequentially")
print("   ├── └── Return success/failure")
print()

print("✅ Benefits of this approach:")
print("   ✅ Simple and direct function call")
print("   ✅ No complex flags or parameters needed")
print("   ✅ Automatic execution after successful planning")
print("   ✅ Clean separation of concerns")
print("   ✅ Easy to integrate into larger systems")
print()

print("🎯 Result:")
print("   The planner now seamlessly integrates with robot execution!")
print("   Just call max_planner.py and the robot will automatically execute the plan.") 