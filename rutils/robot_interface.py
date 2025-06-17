"""High-level interface for Franka robot operations - CORE INTERFACE ONLY."""

import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .config import Config
from .selenium_helper import SeleniumHelper
from .locators import FrankaLocators

class FrankaRobotInterface:
    """Core robot interface for authentication, status, and basic operations."""
    
    def __init__(self, driver: webdriver.Chrome, config: Config, logger: logging.Logger):
        self.driver = driver
        self.config = config
        self.logger = logger
        self.selenium = SeleniumHelper(driver, config, logger)
        self.locators = FrankaLocators()
        # Add commands instance for utility methods
        self._commands = None
        
    def is_dashboard_loaded(self) -> bool:
        """Check if we're in the dashboard by looking for task list."""
        # Look for specific tasks that should be in the task list
        task_indicators = [
            (By.XPATH, "//span[text()='Gripper_open']"),
            (By.XPATH, "//span[text()='Gripper_close']"),
            (By.XPATH, "//span[text()='Move_robot']"),
            (By.XPATH, "//*[contains(text(), 'Gripper_open')]"),
            (By.XPATH, "//*[contains(text(), 'Gripper_close')]"),
            (By.XPATH, "//*[contains(text(), 'Move_robot')]"),
            # Also check for library section
            (By.CSS_SELECTOR, "one-library"),
            (By.XPATH, "//one-library"),
        ]
        
        task_elem = self.selenium.try_multiple_locators(task_indicators, timeout=1)
        if task_elem:
            self.logger.debug("✅ Dashboard detected - found task list")
            return True
        
        self.logger.debug("❌ Dashboard not detected - no task list found")
        return False
    
    def debug_page_status(self) -> None:
        """Quick debug of robot status."""
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # Look for key status indicators
            status_keywords = ["Ready", "Joints locked", "Robot not operational", "not operational"]
            found = [kw for kw in status_keywords if kw in page_text]
            
            if found:
                self.logger.info(f"🔍 Status keywords found: {found}")
            else:
                self.logger.info("🔍 No clear status found")
                
        except Exception as e:
            self.logger.debug(f"Debug failed: {e}")
    
    def has_control_conflict(self) -> bool:
        """Check for control conflict dialog."""
        conflict_indicators = [
            "has control",
            "Another user",
            "control conflict",
            "already controlling"
        ]
        
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            for indicator in conflict_indicators:
                if indicator in page_text:
                    return True
        except:
            # Fallback to element search
            conflict_elem = self.selenium.try_multiple_locators([
                (By.XPATH, "//*[contains(text(), 'has control')]"),
                (By.XPATH, "//*[contains(text(), 'Another user')]"),
            ], timeout=1)
            return conflict_elem is not None
        
        return False

    @property
    def commands(self):
        """Lazy initialization of commands instance to avoid circular import."""
        if self._commands is None:
            from .robot_commands import FrankaRobotCommands
            self._commands = FrankaRobotCommands(self, self.logger)
        return self._commands

    def stop_any_running_task(self) -> bool:
        """Check if there's a running task and stop it during initialization."""
        self.logger.info("🔍 Checking for any running tasks...")
        
        try:
            # Use the same locators as click_stop_button
            stop_button_locators = [
                (By.XPATH, "/html/body/div[2]/section/one-sidebar/div[2]/div/div[2]/footer/section/div/div"),
                (By.XPATH, "/html/body/div[2]/section/one-sidebar/div[1]/div/div[2]/footer/section/div/div"),
                (By.CSS_SELECTOR, "one-sidebar footer section div div"),
                (By.XPATH, "//footer//section//div//div[contains(@class, 'execution-button')]"),
            ]
            
            stop_button = self.selenium.try_multiple_locators(stop_button_locators, timeout=2)
            if stop_button:
                # Check if it's actually a stop button (task running)
                button_html = stop_button.get_attribute('outerHTML') or ""
                button_classes = stop_button.get_attribute("class") or ""
                
                if ("stop" in button_html.lower() or "cancel" in button_html.lower() or
                    "stop" in button_classes.lower()):
                    self.logger.info("🛑 Found running task - stopping it...")
                    # Use the existing click_stop_button from commands
                    return self.commands.click_stop_button()
                else:
                    self.logger.info("✅ No running tasks detected")
                    return True
            else:
                self.logger.info("✅ No running tasks detected")
                return True
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking for running tasks: {e}")
            return True  # Continue anyway
    
    def navigate_and_login(self) -> None:
        """Navigate to robot interface and handle authentication efficiently."""
        self.logger.info(f"🌐 Navigating to {self.config.robot_url}")
        self.driver.get(self.config.robot_url)
        
        # Give page time to load
        time.sleep(3)
        
        # Check if we're already in dashboard
        if self.is_dashboard_loaded():
            self.logger.info("✅ Already in dashboard - no login needed")
        else:
            # Look for password field to determine if login is needed
            password_field = self.selenium.try_multiple_locators(
                self.locators.LOGIN_PASSWORD, timeout=3
            )
            
            if password_field:
                self.logger.info("🔐 Performing login...")
                
                # Find and fill username
                username_field = self.selenium.try_multiple_locators(self.locators.LOGIN_USERNAME, timeout=2)
                if username_field:
                    username_field.clear()
                    username_field.send_keys(self.config.username)
                
                # Fill password
                password_field.clear()
                password_field.send_keys(self.config.password)
                
                # Submit
                submit_button = self.selenium.try_multiple_locators(self.locators.LOGIN_SUBMIT, timeout=2)
                if submit_button:
                    self.selenium.click_element_robust(submit_button)
                
                # Wait for login to complete
                time.sleep(3)
                self.logger.info("✅ Login successful")
            else:
                self.logger.info("✅ No login required")
        
        # Check for control conflicts
            if self.has_control_conflict():
                self.logger.error("❌ Another user has control of the robot!")
                raise RuntimeError("❌ Cannot obtain robot control")
            
            self.logger.info("✅ Robot control confirmed")
            
            # Wait for robot status to load
            self.wait_for_status_load()
    
    def check_robot_status(self) -> dict:
        """Check current robot status - Ready means everything is good."""
        self.logger.info("🔍 Checking robot status...")
        
        status = {
            'joints_locked': False,
            'ready': False,
            'not_operational': False
        }
        
        # Wait a bit for status to load if needed
        for attempt in range(10):  # Max 3 seconds
            try:
                page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                
                # Ready status means everything is good - joints are automatically unlocked
                if "ready" in page_text:
                    status['ready'] = True
                    self.logger.info("✅ Robot is READY")
                    return status
                
                # If not ready, check what the issue is
                if "joints locked" in page_text:
                    status['joints_locked'] = True
                    self.logger.info("🔒 Joints are LOCKED")
                    return status
                
                if "robot not operational" in page_text or "not operational" in page_text:
                    status['not_operational'] = True
                    if attempt < 5:
                        self.logger.info(f"⚠️ Robot not operational - waiting... ({attempt + 1}/6)")
                        time.sleep(0.5)
                        continue
                    else:
                        self.logger.info("⚠️ Robot not operational")
                        return status
                
            except Exception as e:
                self.logger.debug(f"Status check failed: {e}")
                time.sleep(0.5)
        
        self.logger.warning("⚠️ Could not determine robot status")
        return status

    def unlock_joints(self) -> bool:
        """Unlock robot joints by clicking the brake OPEN button and confirming."""
        self.logger.info("🔓 Attempting to unlock joints...")
        
        # First find the brake button
        brake_open_locators = [
            (By.XPATH, "//li[contains(@data-bind, 'switchBrakes.bind')]"),
            (By.XPATH, "//li[contains(@data-bind, 'switchBrakes')]"),
            (By.XPATH, "//li[contains(@data-bind, 'brakes.states.open')]"),
            (By.XPATH, "//li[@data-bind='click: robot.brakes.switchBrakes.bind(robot.brakes, brakes.states.open)']"),
        ]
        
        brake_open_button = self.selenium.try_multiple_locators(brake_open_locators, timeout=3)
        if not brake_open_button:
            self.logger.error("❌ Could not find brake OPEN button")
            return False
        
        self.logger.info("🔓 Found brake button - clicking...")
        if not self.selenium.click_element_robust(brake_open_button):
            self.logger.error("❌ Failed to click brake button")
            return False
        
        # Wait for and handle confirmation dialog
        self.logger.info("🔍 Waiting for confirmation dialog...")
        
        confirm_open_locators = [
            (By.XPATH, "//button[text()='Open']"),
            (By.XPATH, "//button[contains(text(), 'Open')]"),
            (By.XPATH, "//button[contains(@class, 'btn') and contains(text(), 'Open')]"),
            (By.XPATH, "//input[@type='button' and @value='Open']"),
            (By.XPATH, "//input[@type='submit' and @value='Open']"),
        ]
        
        # Wait up to 5 seconds for dialog
        confirm_button = None
        for _ in range(10):
            confirm_button = self.selenium.try_multiple_locators(confirm_open_locators, timeout=0.5)
            if confirm_button:
                break
            time.sleep(0.5)
        
        if not confirm_button:
            self.logger.error("❌ Confirmation dialog did not appear")
            return False
        
        self.logger.info("✅ Found confirmation dialog - clicking 'Open' button...")
        if not self.selenium.click_element_robust(confirm_button):
            self.logger.error("❌ Failed to click confirmation button")
            return False
        
        self.logger.info("✅ Clicked confirmation - unlock process initiated")
        
        # Give the unlock process a moment to start
        time.sleep(1)
        
        return True
        
    def wait_for_status_load(self) -> None:
        """Wait for initial robot status to load after navigation."""
        self.logger.info("⏳ Waiting for robot status to load...")
        
        for attempt in range(10):  # 10 seconds max
            try:
                page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                
                # Check if we have any meaningful status
                if any(status in page_text for status in ["ready", "joints locked", "not operational", "robot not operational"]):
                    self.logger.info("✅ Robot status loaded")
                    return
                    
            except Exception as e:
                self.logger.debug(f"Status load check failed: {e}")
                
            time.sleep(1)
        
        self.logger.warning("⚠️ Timeout waiting for status to load")

    def wait_for_ready(self) -> None:
        """Wait for robot Ready status - simple check."""
        self.logger.info("⏳ Waiting for robot to be ready...")
        
        for attempt in range(30):  # 30 seconds max
            try:
                page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                
                if "ready" in page_text:
                    self.logger.info("🎯 Robot is READY! 🎉")
                    return
                
                # Only log every 5 seconds to reduce spam
                if attempt % 5 == 0:
                    if "joints locked" in page_text:
                        self.logger.info(f"🔒 Joints locked, waiting... ({attempt}s)")
                    elif "not operational" in page_text:
                        self.logger.info(f"⚠️ Not operational, waiting... ({attempt}s)")
                    else:
                        self.logger.info(f"⏳ Waiting for Ready... ({attempt}s)")
                    
            except Exception as e:
                self.logger.debug(f"Ready check failed: {e}")
                
            time.sleep(1)
        
        self.logger.warning("⚠️ Timeout waiting for Ready status")
            
    def ensure_joints_unlocked(self) -> None:
        """Ensure robot is ready - simple and direct."""
        
        # Stop any running tasks
        self.stop_any_running_task()
        
        # Check current status
        status = self.check_robot_status()
        
        # If ready, we're done!
        if status['ready']:
            self.logger.info("🎯 Robot is READY! 🎉")
            return
        
        # If joints are locked, unlock them
        if status['joints_locked']:
            self.logger.info("🔒 Joints locked - unlocking...")
            
            if not self.unlock_joints():
                self.logger.error("❌ Failed to unlock robot")
                raise RuntimeError("Failed to unlock robot joints")
            
            # Wait for ready after unlock
            self.wait_for_ready()
            return
        
        # If not operational or unclear status, just wait for ready
        if status['not_operational']:
            self.logger.info("⚠️ Robot not operational - waiting for ready...")
        else:
            self.logger.info("🤔 Robot status unclear - waiting for ready...")
        
        self.wait_for_ready()
    
    def release_control(self) -> None:
        """Release control of the robot safely."""
        try:
            self.logger.info("🔓 Releasing robot control...")
            # Set a very short timeout for this operation
            self.driver.set_page_load_timeout(2)
            self.driver.execute_script("window.location.reload();")
            time.sleep(1)  # Short wait for reload to start
        except Exception as e:
            self.logger.debug(f"Could not release control cleanly: {e}")
            # Don't log as warning since this is expected when browser closes