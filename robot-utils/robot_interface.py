"""High-level interface for Franka robot operations - CORE INTERFACE ONLY."""

import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
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
    
    def is_dashboard_loaded(self) -> bool:
        """Quick check if we're already in the dashboard using 'Pilot'."""
        pilot_elem = self.selenium.try_multiple_locators([
            (By.XPATH, "//*[contains(text(), 'Pilot')]"),
            (By.XPATH, "//*[contains(text(), 'PILOT')]"),
        ], timeout=2)
        return pilot_elem is not None
    
    def has_control_conflict(self) -> bool:
        """Quick check for control conflict dialog."""
        conflict_elem = self.selenium.try_multiple_locators([
            (By.XPATH, "//*[contains(text(), 'has control')]"),
            (By.XPATH, "//*[contains(text(), 'Another user')]"),
        ], timeout=1)
        return conflict_elem is not None
    
    def navigate_and_login(self) -> None:
        """Navigate to robot interface and handle authentication efficiently."""
        self.logger.info(f"🌐 Navigating to {self.config.robot_url}")
        self.driver.get(self.config.robot_url)
        time.sleep(3)
        
        if self.is_dashboard_loaded():
            self.logger.info("✅ Already in dashboard - no login needed")
        else:
            password_field = self.selenium.try_multiple_locators(
                self.locators.LOGIN_PASSWORD, timeout=3
            )
            
            if password_field:
                self.logger.info("🔐 Performing login...")
                
                username_field = self.selenium.try_multiple_locators(self.locators.LOGIN_USERNAME, timeout=2)
                if username_field:
                    username_field.clear()
                    username_field.send_keys(self.config.username)
                
                password_field.clear()
                password_field.send_keys(self.config.password)
                
                submit_button = self.selenium.try_multiple_locators(self.locators.LOGIN_SUBMIT, timeout=2)
                if submit_button:
                    self.selenium.click_element_robust(submit_button)
                
                time.sleep(3)
                self.logger.info("✅ Login successful")
            else:
                self.logger.info("✅ No login required")
        
        if self.has_control_conflict():
            self.logger.error("❌ Another user has control of the robot!")
            raise RuntimeError("❌ Cannot obtain robot control")
        
        self.logger.info("✅ Robot control confirmed")
    
    def check_robot_status(self) -> dict:
        """Check current robot status efficiently."""
        self.logger.info("🔍 Checking robot status...")
        
        status = {
            'joints_locked': False,
            'end_effector_initialized': True,
            'ready': False
        }
        
        # Check for "Ready" status first
        ready_elem = self.selenium.try_multiple_locators([
            (By.XPATH, "//*[contains(text(), 'Ready')]"),
            (By.XPATH, "//*[contains(text(), 'READY')]"),
        ], timeout=2)
        
        if ready_elem:
            status['ready'] = True
            self.logger.info("✅ Robot is READY")
            return status
        
        # Check for "Joints locked" text
        joints_locked_elem = self.selenium.try_multiple_locators([
            (By.XPATH, "//*[contains(text(), 'Joints locked')]"),
            (By.XPATH, "//*[contains(text(), 'joints locked')]"),
        ], timeout=2)
        
        if joints_locked_elem:
            status['joints_locked'] = True
            self.logger.info("🔒 Joints are LOCKED")
        else:
            self.logger.info("🔓 Joints are unlocked")
        
        # Check for "End-effector not initialized" text
        end_effector_elem = self.selenium.try_multiple_locators([
            (By.XPATH, "//*[contains(text(), 'End-effector not initialized')]"),
            (By.XPATH, "//*[contains(text(), 'not initialized')]"),
        ], timeout=2)
        
        if end_effector_elem:
            status['end_effector_initialized'] = False
            self.logger.info("⚠️ End-effector NOT initialized")
        else:
            self.logger.info("✅ End-effector initialized")
        
        return status
    
    def unlock_joints(self) -> bool:
        """Unlock robot joints by clicking the brake OPEN button."""
        self.logger.info("🔓 Attempting to unlock joints...")
        
        brake_open_locators = [
            (By.XPATH, "//li[contains(@data-bind, 'switchBrakes.bind')]"),
            (By.XPATH, "//li[contains(@data-bind, 'switchBrakes')]"),
            (By.XPATH, "//li[contains(@data-bind, 'brakes.states.open')]"),
        ]
        
        brake_open_button = self.selenium.try_multiple_locators(brake_open_locators, timeout=3)
        if brake_open_button:
            self.logger.info("🔓 Clicking brake OPEN button...")
            self.selenium.click_element_robust(brake_open_button)
            
            # Handle confirmation dialog
            self.logger.info("🔍 Looking for confirmation dialog...")
            time.sleep(2)
            
            confirm_open_locators = [
                (By.XPATH, "//button[text()='Open']"),
                (By.XPATH, "//button[contains(text(), 'Open')]"),
                (By.XPATH, "//input[@value='Open']"),
                (By.XPATH, "//*[text()='Open' and (self::button or self::input)]"),
            ]
            
            confirm_button = self.selenium.try_multiple_locators(confirm_open_locators, timeout=5)
            if confirm_button:
                self.logger.info("✅ Found confirmation dialog - clicking 'Open' button...")
                self.selenium.click_element_robust(confirm_button)
                time.sleep(2)
            else:
                self.logger.warning("⚠️ No confirmation dialog found - continuing...")
            
            self.logger.info("⏳ Unlock process initiated - waiting for completion...")
            time.sleep(3)
            return True
        else:
            self.logger.warning("⚠️ Could not find brake OPEN button")
            return False
    
    def wait_for_ready(self) -> None:
        """Wait for robot to reach ready state by checking for 'Ready' text."""
        self.logger.info("⏳ Waiting for robot to be ready...")
        
        for attempt in range(20):
            ready_elem = self.selenium.try_multiple_locators([
                (By.XPATH, "//*[contains(text(), 'Ready')]"),
                (By.XPATH, "//*[contains(text(), 'READY')]"),
            ], timeout=1)
            
            if ready_elem:
                self.logger.info("🎯 Robot is READY! 🎉")
                break
            
            # Check blocking issues
            blocking_issues = []
            
            joints_locked = self.selenium.try_multiple_locators([
                (By.XPATH, "//*[contains(text(), 'Joints locked')]"),
            ], timeout=1)
            if joints_locked:
                blocking_issues.append("Joints locked")
            
            not_initialized = self.selenium.try_multiple_locators([
                (By.XPATH, "//*[contains(text(), 'not initialized')]"),
            ], timeout=1)
            if not_initialized:
                blocking_issues.append("End-effector not initialized")
            
            if blocking_issues:
                self.logger.info(f"⏳ Still waiting... Issues: {', '.join(blocking_issues)} (attempt {attempt + 1}/20)")
            else:
                self.logger.info(f"⏳ Waiting for Ready status... (attempt {attempt + 1}/20)")
            
            time.sleep(2)
        else:
            self.logger.warning("⚠️ Timeout waiting for robot Ready status")
        
        time.sleep(2)
        self.logger.info("✅ Robot ready check completed")
    
    def ensure_joints_unlocked(self) -> None:
        """Ensure robot joints are unlocked and robot is ready."""
        status = self.check_robot_status()
        
        if status['ready']:
            self.logger.info("🎯 Robot is already READY! 🎉")
            return
        
        if status['joints_locked']:
            self.logger.info("🔒 Joints are locked - attempting to unlock...")
            if self.unlock_joints():
                self.wait_for_ready()
                
                final_status = self.check_robot_status()
                if not final_status['ready']:
                    if final_status['joints_locked']:
                        self.logger.error("❌ Joints still locked after unlock attempt")
                        raise RuntimeError("Failed to unlock robot joints")
                    else:
                        self.logger.warning("⚠️ Robot unlocked but not showing ready status")
            else:
                self.logger.error("❌ Failed to click unlock button")
                raise RuntimeError("Failed to unlock robot joints")
        else:
            self.logger.info("✅ Joints already unlocked")
            self.wait_for_ready()
    
    def release_control(self) -> None:
        """Release control of the robot."""
        try:
            self.logger.info("🔓 Releasing robot control...")
            self.driver.execute_script("window.location.reload();")
            time.sleep(2)
        except Exception as e:
            self.logger.warning(f"Could not release control cleanly: {e}")