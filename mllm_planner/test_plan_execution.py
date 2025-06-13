#!/usr/bin/env python3
"""
Test script for robot plan execution functionality.
Demonstrates the new plan string parsing and execution features.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rutils.robot_action import FrankaAutomation
from rutils.config import Config

def test_plan_parsing():
    """Test the plan parsing functionality without robot execution."""
    
    # Create configuration (these values won't be used for parsing test)
    config = Config(
        robot_ip="172.16.0.2",
        local_ip="172.16.0.1", 
        network_interface="enp2s0"
    )
    
    # Create automation instance
    automation = FrankaAutomation(config)
    
    # Test plan strings
    test_plans = [
        "plan: (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 40), move)",
        "plan: (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 30), move) -> (USB Drive, (25, 25, 25), move) -> (Pen, (25, 25, 45), move)",
        "plan: (Cup, (10.5, -5.2, 35.7), move) -> (Cup, (10.5, -5.2, 60.7), move) -> (Table, (50, 20, 30), move) -> (Cup, (50, 50, 30), move)"
    ]
    
    print("🧪 Testing Plan Parsing Functionality")
    print("=" * 50)
    
    for i, plan in enumerate(test_plans, 1):
        print(f"\n📋 Test Plan {i}:")
        print(f"   {plan}")
        print("\n🔍 Parsed Actions:")
        
        parsed_actions = automation.parse_plan_string(plan)
        
        for j, (obj, coords, action) in enumerate(parsed_actions, 1):
            x, y, z = coords
            print(f"   {j}. {action.upper()} {obj} at position ({x}, {y}, {z})")
        
        if not parsed_actions:
            print("   ❌ No actions parsed!")
    
    print("\n✅ Plan parsing test completed!")

def main():
    """Main function to run tests or provide usage examples."""
    
    print("🤖 Robot Plan Execution Test & Demo")
    print("=" * 40)
    
    # Test plan parsing functionality
    test_plan_parsing()
    
    print("\n📖 Usage Examples:")
    print("=" * 40)
    
    print("\n1. Execute a simple move and pick plan:")
    print('   python rutils/robot_action.py --plan "plan: (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 30), pick)"')
    
    print("\n2. Execute a complex pick and place plan:")
    print('   python rutils/robot_action.py --plan "plan: (Pen, (15, 20, 30), move) -> (Pen, (15, 20, 30), pick) -> (USB Drive, (25, 25, 25), move) -> (Pen, (25, 25, 25), place)"')
    
    print("\n3. Execute plan in headless mode:")
    print('   python rutils/robot_action.py --headless --plan "plan: (Cup, (10, 10, 30), move) -> (Cup, (10, 10, 30), pick)"')
    
    print("\n4. Execute plan without network setup (if already configured):")
    print('   python rutils/robot_action.py --no-network-setup --plan "plan: (Object, (0, 0, 0), move)"')
    
    print("\n📝 Plan Format Rules:")
    print("=" * 40)
    print("• Format: 'plan: (Object, (x, y, z), action) -> (Object, (x, y, z), action) -> ...'")
    print("• Object: Name of the object to manipulate")
    print("• (x, y, z): 3D coordinates where robot should move")
    print("• Actions:")
    print("  - move: Move robot to specified coordinates")
    print("  - pick: Activate suction (pick up object)")
    print("  - place: Deactivate suction (place object)")
    print("• Robot waits 3 seconds between each action for stability")
    
    print("\n✨ Enhanced Features:")
    print("=" * 40)
    print("• ✅ Plan string parsing with error handling")
    print("• ✅ Sequential action execution with delays")
    print("• ✅ Robot movement integration")
    print("• ✅ Suction control (pick/place)")
    print("• ✅ Comprehensive logging")
    print("• ✅ Graceful error handling")

if __name__ == "__main__":
    main() 