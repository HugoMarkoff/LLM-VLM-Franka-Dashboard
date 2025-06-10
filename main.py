#!/usr/bin/env python3
"""
Franka Desk Selenium Automation Suite - Modular Version
======================================================

A robust automation framework for Franka Desk robot interface.

Usage:
    python main.py --init-only          # Initialize robot, keep browser open
    python main.py --init-only --interactive # Interactive shell with robot commands
    python main.py --headless           # Full automation headless
    python main.py --config-only        # Configure tasks only
"""

import sys
import time
import argparse
import subprocess
from typing import Optional

# Selenium imports
from selenium import webdriver

# Local imports
from utils.config import Config
from utils.logger import setup_logging
from utils.network_manager import NetworkManager
from utils.chrome_manager import ChromeDriverManager
from utils.robot_interface import FrankaRobotInterface
from utils.robot_commands import FrankaRobotCommands
from utils.signal_handler import GracefulKiller


def run_interactive_shell(automation: 'FrankaAutomation'):
    """Run interactive Python shell with robot commands available."""
    import code
    
    # Create convenient command shortcuts
    def config_open(speed=20):
        return automation.gripper_open_config(speed)
    
    def config_close(speed=50, force=80, load=400):
        return automation.gripper_close_config(speed, force, load)
    
    def open_gripper():
        return automation.gripper_open()
    
    def close_gripper():
        return automation.gripper_close()
    
    def move_robot(x=0, y=0, z=0, speed=5, acceleration=5):
        return automation.commands.move_robot(x, y, z, speed, acceleration)
    
    def robot_status():
        return automation.robot.check_robot_status()
    
    # Create interactive namespace
    interactive_locals = {
        'robot': automation,
        'config_open': config_open,
        'config_close': config_close, 
        'open_gripper': open_gripper,
        'close_gripper': close_gripper,
        'move_robot': move_robot,
        'robot_status': robot_status,
        'automation': automation,
    }
    
    print("\n" + "="*60)
    print("🤖 FRANKA ROBOT INTERACTIVE SHELL")
    print("="*60)
    print("Available commands:")
    print("  config_open(speed=20)                    - Configure gripper open")
    print("  config_close(speed=50, force=80, load=400) - Configure gripper close") 
    print("  open_gripper()                           - Execute gripper open")
    print("  close_gripper()                          - Execute gripper close")
    print("  move_robot(x=0, y=0, z=0, speed=5, accel=5) - Move robot relative")
    print("  robot_status()                           - Check robot status")
    print("  robot                                    - Access full robot instance")
    print("\nParameter ranges:")
    print("  Gripper speed: 10-100    (speed %)")
    print("  Gripper force: 20-100    (grasping force in Newtons)")
    print("  Gripper load:  10-1000   (load capacity in grams)")
    print("  Robot speed:   5-100     (movement speed %)")
    print("  Robot accel:   5-100     (acceleration %)")
    print("  Robot x,y,z:   any float (relative movement in mm)")
    print("\nExamples:")
    print("  config_open(30)                 # Set open speed to 30%")
    print("  config_close(60, 85, 500)      # Set close: 60% speed, 85N force, 500g load")
    print("  config_close()                  # Use defaults (50% speed, 80N force, 400g load)")
    print("  open_gripper()                  # Open gripper")
    print("  close_gripper()                 # Close gripper")
    print("  move_robot(10, 0, 5)            # Move +10mm X, +5mm Z")
    print("  move_robot(-5, 10, 0, 10, 15)   # Move -5mm X, +10mm Y, 10% speed, 15% accel")
    print("  move_robot(z=-20)               # Move -20mm Z only")
    print("  exit()                          # Exit shell")
    print("="*60)
    
    # Start interactive shell
    try:
        code.interact(local=interactive_locals, banner="")
    except (EOFError, KeyboardInterrupt):
        print("\n🛑 Exiting interactive shell...")


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
    
    def keep_alive(self, interactive: bool = False) -> None:
        """Keep the robot session alive until user interrupts."""
        if not self.is_robot_ready():
            self.logger.error("❌ No active robot session to keep alive")
            return
        
        if interactive:
            # Drop into interactive Python shell
            run_interactive_shell(self)
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Franka Desk Selenium Automation - Modular Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Full automation with GUI
  python main.py --headless               # Headless automation
  python main.py --init-only              # Initialize robot only, keep UI open
  python main.py --init-only --interactive # Interactive shell with robot commands
  python main.py --config-only            # Configure tasks only

Interactive shell usage:
  python main.py --init-only --interactive
  >>> config_open(30)        # Configure gripper open speed
  >>> open_gripper()         # Execute gripper open
  >>> close_gripper()        # Execute gripper close
  >>> exit()                 # Exit shell
        """
    )
    
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--init-only", action="store_true", help="Initialize robot only, then keep UI open")
    parser.add_argument("--interactive", action="store_true", help="Drop into interactive Python shell (use with --init-only)")
    parser.add_argument("--config-only", action="store_true", help="Only configure tasks, don't execute them")
    parser.add_argument("--setup-network", action="store_true", help="Set up network configuration and test connectivity")
    parser.add_argument("--check-versions", action="store_true", help="Check Chrome and ChromeDriver versions")
    parser.add_argument("--no-network-setup", action="store_true", help="Skip network configuration")
    parser.add_argument("--robot-ip", default="172.16.0.2", help="Robot IP address (default: 172.16.0.2)")
    parser.add_argument("--local-ip", default="172.16.0.1", help="Local IP address (default: 172.16.0.1)")
    parser.add_argument("--interface", default="enp2s0", help="Network interface (default: enp2s0)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        robot_ip=args.robot_ip,
        local_ip=args.local_ip,
        network_interface=args.interface
    )
    
    logger = setup_logging()
    
    # Handle version check
    if args.check_versions:
        chrome_manager = ChromeDriverManager(config, logger)
        try:
            import subprocess
            import re
            
            commands = [
                ["google-chrome", "--version"],
                ["google-chrome-stable", "--version"],
                ["chromium-browser", "--version"],
                ["chromium", "--version"],
            ]
            
            chrome_version = None
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version_match = re.search(r'(\d+)\.(\d+)\.(\d+)\.(\d+)', result.stdout)
                        if version_match:
                            chrome_version = version_match.group(0)
                            break
                except:
                    continue
            
            if chrome_version:
                logger.info(f"🌍 Chrome version: {chrome_version}")
            else:
                logger.warning("⚠️ Could not determine Chrome version")
            
            driver_path = chrome_manager._get_chromedriver_auto()
            logger.info(f"📍 Auto-managed ChromeDriver path: {driver_path}")
            logger.info("✅ ChromeDriver auto-management working")
            
        except Exception as e:
            logger.error(f"❌ ChromeDriver auto-management failed: {e}")
        
        return 0
    
    # Handle network setup only
    if args.setup_network:
        network_manager = NetworkManager(config, logger)
        
        if network_manager.configure_network():
            network_manager.test_robot_connectivity()
            return 0
        else:
            return 1
    
    # Run automation
    automation = FrankaAutomation(config)
    
    success = automation.run_automation(
        headless=args.headless,
        init_only=args.init_only,
        config_only=args.config_only,
        interactive=args.interactive,
        setup_network=not args.no_network_setup
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())