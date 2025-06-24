#!/usr/bin/env python3
"""
Franka Desk Selenium Automation Suite - Modular Version
======================================================

Args from planner: plan: (Object, (x, y, z), action) -> (Object, (15, 20, 30), action) -> ...

"""

import sys
import time
import argparse
import subprocess
import re
from typing import Optional, List, Tuple
from pathlib import Path

# Add parent directory to Python path to find rutils
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Selenium imports
from selenium import webdriver

# Local imports
from rutils.config import Config
from rutils.logger import setup_logging
from rutils.network_manager import NetworkManager
from rutils.chrome_manager import ChromeDriverManager
from rutils.robot_interface import FrankaRobotInterface
from rutils.robot_commands import FrankaRobotCommands
from rutils.signal_handler import GracefulKiller


class FrankaAutomation:
    """Main automation application with persistent browser support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()
        self.network_manager = NetworkManager(config, self.logger)
        self.chrome_manager = ChromeDriverManager(config, self.logger)
        self.driver: Optional[webdriver.Chrome] = None
        self.robot: Optional[FrankaRobotInterface] = None
        self.commands: Optional[FrankaRobotCommands] = None
        self.killer: Optional[GracefulKiller] = None
        self._is_initialized = False
    
    def start_robot(self, headless: bool = False, setup_network: bool = True) -> bool:
        """Initialize robot and keep browser session active for future calls."""
        try:
            # Environment setup
            if setup_network and not self.setup_environment(setup_network):
                self.logger.error("❌ Environment setup failed")
                return False
            
            # Create persistent browser session
            if not self.driver:
                self.driver = self.chrome_manager.create_driver(headless)
                self.robot = FrankaRobotInterface(self.driver, self.config, self.logger)
                self.commands = FrankaRobotCommands(self.robot, self.logger)
                self.killer = GracefulKiller(self)
            
            # Initialize robot
            self.robot.navigate_and_login()
            self.robot.ensure_joints_unlocked()
            
            self._is_initialized = True
            self.logger.info("🎯 Robot ready for commands! Use individual functions or Ctrl+C to exit.")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Robot initialization failed: {e}")
            self.save_debug_info()
            return False
    def get_current_pos(self) -> dict:
        """Get current robot position - wrapper for interactive use."""
        if not self.is_robot_ready():
            return None
        
        try:
            return self.commands.get_current_position()
        except Exception as e:
            self.logger.error(f"❌ Get position failed: {e}")
            return None

    def move_to_home(self) -> bool:
        """Move robot to home position - wrapper for interactive use."""
        if not self.is_robot_ready():
            return False
        
        try:
            return self.commands.move_to_home_position()
        except Exception as e:
            self.logger.error(f"❌ Move to home failed: {e}")
            return False
    
    def is_robot_ready(self) -> bool:
        """Check if robot is initialized and ready."""
        if not self._is_initialized or not self.driver or not self.robot:
            self.logger.warning("⚠️ Robot not initialized. Call start_robot() first.")
            return False
        
        # Quick browser health check
        try:
            _ = self.driver.current_url
            return True
        except:
            self.logger.error("❌ Browser session lost. Need to restart robot.")
            self._is_initialized = False
            return False
    
    # ========== GRIPPER COMMAND DELEGATION ==========
    
    def gripper_open_config(self, speed: int = 40) -> bool:
        """Configure gripper open settings."""
        if not self.is_robot_ready():
            return False
        
        try:
            return self.commands.configure_gripper_open(speed=speed)
        except Exception as e:
            self.logger.error(f"❌ Gripper open config failed: {e}")
            return False
    
    def gripper_close_config(self, speed: int = 50, force: int = 80, load: int = 400) -> bool:
        """Configure gripper close settings."""
        if not self.is_robot_ready():
            return False
        
        try:
            return self.commands.configure_gripper_close(speed=speed, force=force, load=load)
        except Exception as e:
            self.logger.error(f"❌ Gripper close config failed: {e}")
            return False
    
    def gripper_open(self) -> bool:
        """Execute gripper open."""
        if not self.is_robot_ready():
            return False
        
        try:
            return self.commands.gripper_open()
        except Exception as e:
            self.logger.error(f"❌ Gripper open failed: {e}")
            return False
    
    def gripper_close(self) -> bool:
        """Execute gripper close."""
        if not self.is_robot_ready():
            return False
        
        try:
            return self.commands.gripper_close()
        except Exception as e:
            self.logger.error(f"❌ Gripper close failed: {e}")
            return False
    
    # ========== SESSION MANAGEMENT ==========
    
    def stop_robot(self) -> None:
        """Clean shutdown of robot session."""
        self.logger.info("🛑 Stopping robot session...")
        self.cleanup()
        self._is_initialized = False

    def suction_on(self, load: int = None, vacuum: int = None, timeout: float = None) -> bool:
        """Execute suction_on command. If parameters provided, configure first."""
        if not self.is_robot_ready():
            return False
        
        try:
            return self.commands.suction_on(load=load, vacuum=vacuum, timeout=timeout)
        except Exception as e:
            self.logger.error(f"❌ Suction_on failed: {e}")
        return False
    
    def suction_off(self) -> bool:
        """Execute suction_off command."""
        if not self.is_robot_ready():
            return False
        
        try:
            return self.commands.suction_off()
        except Exception as e:
            self.logger.error(f"❌ Suction_off failed: {e}")
        return False

    def move_robot(self, x: float = 0, y: float = 0, z: float = 0, speed: int = 5, acceleration: int = 5) -> bool:
        """Execute robot movement command."""
        if not self.is_robot_ready():
            return False
        
        try:
            return self.commands.move_robot(x=x, y=y, z=z, speed=speed, acceleration=acceleration)
        except Exception as e:
            self.logger.error(f"❌ Robot movement failed: {e}")
        return False
    
    def parse_plan_string(self, plan_string: str) -> List[Tuple[str, Tuple[float, float, float], str]]:
        """
        Parse plan string into list of actions.
        Format: "plan: (Object, (x, y, z), action) -> (Object, (x, y, z), action) -> ..."
        Returns: List of tuples (object_name, (x, y, z), action)
        """
        self.logger.info(f"🔍 Parsing plan string: {plan_string}")
        
        # Remove "plan: " prefix if present
        if plan_string.startswith("plan: "):
            plan_string = plan_string[6:]
        
        # Split by "->" to get individual actions
        action_strings = plan_string.split(" -> ")
        
        parsed_actions = []
        
        for action_str in action_strings:
            action_str = action_str.strip()
            
            # Pattern to match: (Object, (x, y, z), action)
            pattern = r'\(([^,]+),\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\),\s*([^)]+)\)'
            match = re.match(pattern, action_str)
            
            if match:
                object_name = match.group(1).strip()
                x = float(match.group(2).strip())
                y = float(match.group(3).strip())
                z = float(match.group(4).strip())
                action = match.group(5).strip()
                
                parsed_actions.append((object_name, (x, y, z), action))
                self.logger.info(f"  ✅ Parsed: {object_name} at ({x}, {y}, {z}) -> {action}")
            else:
                self.logger.error(f"  ❌ Failed to parse action: {action_str}")
        
        return parsed_actions

    def execute_plan(self, plan_string: str) -> bool:
        """
        Execute a plan string by parsing it and executing actions in sequence.
        
        Args:
            plan_string: Plan in format "plan: (Object, (x, y, z), action) -> ..."
            
        Returns:
            bool: True if all actions executed successfully, False otherwise
        """
        if not self.is_robot_ready():
            self.logger.error("❌ Robot not ready for plan execution")
            return False
        
        self.logger.info("🚀 Starting plan execution...")
        
        # Parse the plan string
        parsed_actions = self.parse_plan_string(plan_string)
        
        if not parsed_actions:
            self.logger.error("❌ No valid actions found in plan string")
            return False
        
        # Execute each action in sequence
        for i, (object_name, coordinates, action) in enumerate(parsed_actions, 1):
            x, y, z = coordinates
            self.logger.info(f"🎯 Step {i}/{len(parsed_actions)}: {action} {object_name} at ({x}, {y}, {z})")
            
            success = False
            
            if action.lower() == "pickmove":
                if z >= 500:
                    success = self.move_robot(x, y, 500)
                    success = self.move_robot(0, 0, z-500)
                else:
                    success = self.move_robot(x, y, z)
            
            elif action.lower() == "placemove":
                if z >= 500:
                    success = self.move_robot(x, y, 500)
                    success = self.move_robot(0, 0, z-500)
                else:
                    success = self.move_robot(x, y, z)

                
            elif action.lower() == "pick":
                success = self.suction_on(load=200, vacuum=750, timeout=2.0)
                
            elif action.lower() == "place":
                success = self.suction_off()
            
            elif action.lower() == "home":
                success = self.move_to_home()
                time.sleep(1)
                
            else:
                self.logger.error(f"❌ Unknown action: {action}")
                return False
            
            if not success:
                self.logger.error(f"❌ Failed to execute step {i}: {action} {object_name}")
                return False
        
        self.logger.info("🎉 Plan execution completed successfully!")
        return True
    
    def keep_alive(self, interactive: bool = False) -> None:
        """Keep the robot session alive until user interrupts."""
        if not self.is_robot_ready():
            self.logger.error("❌ No active robot session to keep alive")
            return
        
        if interactive:
            # Drop into interactive Python shell
            # run_interactive_shell(self)
            return
        
        self.logger.info("🔄 Robot session active - Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(1)
                if self.killer and self.killer.kill_now:
                    break
                # Periodic health check
                if not self.is_robot_ready():
                    self.logger.error("❌ Robot session lost")
                    break
        except KeyboardInterrupt:
            self.logger.info("🛑 User requested exit")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.robot and self.driver:
            try:
                self.robot.release_control()
            except Exception:
                pass
        
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
        
        self.chrome_manager.cleanup_all_chrome_processes()
        self.robot = None
        self.commands = None
    
    def setup_environment(self, setup_network: bool = True) -> bool:
        """Set up the environment for robot communication."""
        success = True
        
        if setup_network:
            if not self.network_manager.configure_network():
                success = False
            
            if not self.network_manager.test_robot_connectivity():
                self.logger.warning("Robot connectivity test failed, but continuing...")
        
        return success
    
    def run_automation(self, headless: bool = False, init_only: bool = False, config_only: bool = False, interactive: bool = False, setup_network: bool = True) -> bool:
        """Run the complete automation sequence."""
        try:
            # For init_only, use the new persistent approach
            if init_only:
                if self.start_robot(headless, setup_network):
                    self.keep_alive(interactive=interactive)
                    return True
                return False
            
            # Legacy full automation mode
            if not self.setup_environment(setup_network):
                self.logger.error("❌ Environment setup failed")
                return False
            
            # Create temporary session for full automation
            self.driver = self.chrome_manager.create_driver(headless)
            self.robot = FrankaRobotInterface(self.driver, self.config, self.logger)
            self.commands = FrankaRobotCommands(self.robot, self.logger)
            self.killer = GracefulKiller(self)
            
            try:
                if self.killer and self.killer.kill_now:
                    return False
                
                # Initialize robot
                self.robot.navigate_and_login()
                self.robot.ensure_joints_unlocked()
                
                if self.killer and self.killer.kill_now:
                    return False
                
                # NEW SEQUENCE: Configure and test gripper
                # 1. Configure gripper open speed to 20
                self.commands.configure_gripper_open(speed=20)
                
                if not config_only:
                    if self.killer and self.killer.kill_now:
                        return False
                    
                    # 2. Run gripper open
                    self.commands.gripper_open()
                    
                    # 3. Run gripper close (using default config)
                    self.commands.gripper_close()
                    
                    # 4. Configure gripper close
                    self.commands.configure_gripper_close(speed=50, force=80, load=400)
                
                self.logger.info("🎉 Automation completed successfully!")
                return True
                
            finally:
                self.cleanup()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Automation interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"❌ Automation failed: {e}")
            import traceback
            traceback.print_exc()
            self.save_debug_info()
            return False
    
    def save_debug_info(self) -> None:
        """Save debug information on failure to crashLog folder."""
        if not self.driver:
            return
        
        try:
            from pathlib import Path
            
            crash_dir = Path("crashLog")
            crash_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            screenshot_path = crash_dir / f"franka_debug_{timestamp}.png"
            html_path = crash_dir / f"franka_debug_{timestamp}.html"
            
            self.driver.save_screenshot(str(screenshot_path))
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            
            self.logger.info(f"💾 Debug files saved: {screenshot_path}, {html_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save debug info: {e}")


def execute_plan_from_planner(plan_string: str, headless: bool = False) -> bool:
    """
    Execute a plan string directly from the planner module.
    This function is designed to be called by max_planner.py.
    
    Args:
        plan_string (str): Plan string in format "plan: (Object, (x, y, z), action) -> ..."
        headless (bool): Whether to run browser in headless mode (default: False for GUI)
        
    Returns:
        bool: True if execution successful, False otherwise
    """
    try:
        # Use default configuration
        config = Config(
            robot_ip="172.16.0.2",
            local_ip="172.16.0.1",
            network_interface="enp2s0"
        )
        
        # Create automation instance
        automation = FrankaAutomation(config)
        
        mode_text = "HEADLESS mode" if headless else "GUI mode"
        print(f"🤖 Initializing robot for plan execution in {mode_text}...")
        
        # Initialize robot with configurable headless setting
        if not automation.start_robot(headless=headless, setup_network=True):
            print("❌ Failed to initialize robot")
            return False
        
        print("✅ Robot initialized successfully")
        
        # Execute the plan
        print(f"📋 Executing plan: {plan_string}")
        success = automation.execute_plan(plan_string)
        
        if success:
            print("✅ Plan execution completed successfully")
        else:
            print("❌ Plan execution failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Error in execute_plan_from_planner: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure cleanup always happens
        try:
            print("🧹 Cleaning up robot session...")
            automation.cleanup()
            print("🛑 Robot session cleaned up")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")
