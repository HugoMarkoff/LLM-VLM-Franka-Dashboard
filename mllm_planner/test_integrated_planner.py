#!/usr/bin/env python3
"""
Test script for integrated planner and robot execution.
Demonstrates the new functionality where max_planner.py automatically calls robot_action.py.
"""

import sys
import os
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test JSON inputs
test_scenarios = [
    {
        "name": "Simple Pick Task",
        "json_input": {
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
                    "location": "bottom-right",
                    "object": "paper",
                    "object_center_location": {"x": 30, "y": 10, "z": 25},
                    "object_size": {"width": 20, "length": 30},
                    "object_orientation": 0
                }
            ]
        }
    },
    {
        "name": "Pick and Place Task",
        "json_input": {
            "user_message": "pick up the pen and place it on the paper",
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
                    "action": "place",
                    "location": "bottom-center", 
                    "object": "paper",
                    "object_center_location": {"x": 25, "y": 25, "z": 25},
                    "object_size": {"width": 20, "length": 30},
                    "object_orientation": 0
                }
            ]
        }
    }
]

def test_planning_only():
    """Test planning without execution."""
    print("=" * 60)
    print("🧪 TESTING PLANNING WITH AUTOMATIC ROBOT EXECUTION")
    print("=" * 60)
    print("⚠️  NOTE: The planner now AUTOMATICALLY executes plans on the robot!")
    print("⚠️  Make sure the robot is properly set up and safe to operate.")
    
    response = input("\nProceed with planning and robot execution test? (y/N): ").strip().lower()
    if response != 'y':
        print("🛑 Test skipped by user")
        return
    
    try:
        from mllm_planner.max_planner import robot_action_planner_json
        
        for scenario in test_scenarios:
            print(f"\n📋 Testing: {scenario['name']}")
            print("-" * 40)
            
            json_string = json.dumps(scenario['json_input'])
            print(f"Input: {scenario['json_input']['user_message']}")
            
            try:
                result = robot_action_planner_json(json_string)
                
                if result["success"]:
                    print(f"✅ Planning successful!")
                    print(f"   Task: {result['task_description']}")
                    print(f"   Objects: {result['detected_objects']}")
                    print(f"   Plan: {result['action_plan']}")
                    
                    if result["execution_success"]:
                        print(f"🎉 Robot execution successful!")
                    elif result["execution_success"] is False:
                        print(f"❌ Robot execution failed")
                    else:
                        print(f"⚠️  Robot execution was not attempted")
                else:
                    print(f"❌ Planning failed")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except ImportError as e:
        print(f"❌ Could not import planner: {e}")
        print("   Make sure you're running from the correct directory")

def test_planning_with_execution():
    """This function is no longer needed since execution is automatic."""
    pass

def test_command_line_usage():
    """Show command line usage examples."""
    print("\n" + "=" * 60)
    print("📖 COMMAND LINE USAGE EXAMPLES")
    print("=" * 60)
    
    example_json = json.dumps(test_scenarios[0]['json_input'])
    
    print("\n1. Planning with automatic robot execution:")
    print(f"   python mllm_planner/max_planner.py '{example_json}'")
    
    print("\n2. Direct robot execution (if you already have a plan):")
    print(f"   python mllm_planner/robot_action.py --plan \"plan: (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 30), pick)\"")
    
    print("\n📝 Note: The max_planner.py now automatically executes generated plans on the robot!")

def main():
    """Main test function."""
    print("🚀 SIMPLIFIED INTEGRATED PLANNER AND ROBOT EXECUTION TEST")
    print("=" * 60)
    print("The max_planner.py now automatically calls robot_action.py to execute plans!")
    print("No flags or parameters needed - planning automatically triggers execution.")
    
    # Test planning with automatic execution
    test_planning_only()
    
    # Show command line examples
    test_command_line_usage()
    
    print("\n✅ Integration test completed!")
    print("\n📝 Summary of Simplified Changes:")
    print("  ✅ max_planner.py automatically calls robot_action.py after planning")
    print("  ✅ No --execute flag needed - execution is automatic")
    print("  ✅ Simplified execute_plan_from_planner() function")
    print("  ✅ Direct function call: robot_action.execute_plan_from_planner(plan)")
    print("  ✅ Complete pipeline: Image → Planning → Automatic Robot Execution")

if __name__ == "__main__":
    main() 