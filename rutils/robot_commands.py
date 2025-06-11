"""Robot gripper control commands with real Selenium interactions."""

import time
import logging
from typing import Optional
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .robot_interface import FrankaRobotInterface


class FrankaRobotCommands:
    """Robot gripper control commands with real Selenium automation."""
    
    def __init__(self, robot_interface: FrankaRobotInterface, logger: logging.Logger):
        self.robot = robot_interface
        self.logger = logger
        self.selenium = robot_interface.selenium
        self.driver = robot_interface.driver
    
    def wait_for_element(self, locator, timeout=10):
        """Wait for element to be present and return it."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            return element
        except:
            return None

    def wait_for_clickable_element(self, locator, timeout=10):
        """Wait for element to be clickable and return it."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable(locator)
            )
            return element
        except:
            return None

    # ========== TASK SELECTION ==========
    
    def select_task_from_list(self, task_name: str) -> bool:
        """Select a task from the task list by name."""
        self.logger.info(f"🎯 Selecting task: {task_name}")
        
        task_container_locators = [
            (By.XPATH, "/html/body/div[2]/section/section/one-library/div/div[1]/div[2]"),
            (By.CSS_SELECTOR, "one-library div[class*='div'][class*='div'] div[class*='div']"),
        ]
        
        task_container = self.selenium.try_multiple_locators(task_container_locators, timeout=5)
        if not task_container:
            self.logger.error("❌ Could not find task container")
            return False
        
        task_locators = [
            (By.XPATH, f".//span[text()='{task_name}']"),
            (By.XPATH, f".//*[contains(text(), '{task_name}')]"),
        ]
        
        for locator in task_locators:
            try:
                task_element = task_container.find_element(*locator)
                if task_element:
                    self.selenium.click_element_robust(task_element)
                    self.logger.info(f"✅ Selected task: {task_name}")
                    return True
            except:
                continue
        
        self.logger.error(f"❌ Could not find task: {task_name}")
        return False
    
    # ========== TASK EXECUTION ==========
    
    def click_execution_button(self) -> bool:
        """Click the execution (play) button in the sidebar."""
        self.logger.info("▶️ Clicking execution button...")
        
        execution_locators = [
            (By.CSS_SELECTOR, "body > div:nth-child(2) > section > one-sidebar > div.sidebar-body > div > div.fixed-sections > footer > section > div > div"),
            (By.XPATH, "/html/body/div[2]/section/one-sidebar/div[1]/div/div[2]/footer/section/div/div"),
            (By.CSS_SELECTOR, "one-sidebar footer section div div"),
        ]
        
        execution_button = self.selenium.try_multiple_locators(execution_locators, timeout=5)
        if execution_button:
            self.selenium.click_element_robust(execution_button)
            self.logger.info("✅ Execution button clicked")
            return True
        else:
            self.logger.error("❌ Could not find execution button")
            return False

    def click_confirm_button(self) -> bool:
        """Click the CONFIRM button in the task execution dialog."""
        self.logger.info("✅ Clicking CONFIRM button...")
        
        confirm_locators = [
            (By.XPATH, "/html/body/div[3]/div[3]/div/div[2]/div[3]/div[2]/span/button"),
            (By.XPATH, "//button[contains(., 'CONFIRM')]"),
        ]
        
        confirm_button = self.selenium.try_multiple_locators(confirm_locators, timeout=10)
        if confirm_button:
            self.selenium.click_element_robust(confirm_button)
            self.logger.info("✅ Task execution confirmed")
            return True
        else:
            self.logger.error("❌ Could not find CONFIRM button")
            return False

    def wait_for_ready(self) -> bool:
        """Wait for robot to return to Ready status."""
        self.logger.info("⏳ Waiting for task completion...")
        
        try:
            ready_element = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Ready')]"))
            )
            if ready_element:
                self.logger.info("🎉 Task completed - Robot is Ready!")
                return True
        except TimeoutException:
            self.logger.error("❌ Task timeout")
            return False

    # ========== TASK CONFIGURATION ==========
    
    def click_task_icon_for_config(self) -> bool:
        """Click on the task icon in the programming window to open configuration."""
        self.logger.info("🔧 Clicking task icon to open configuration...")
        
        icon_locators = [
            (By.CSS_SELECTOR, ".drag-area"),
            (By.XPATH, "//div[@class='drag-area']"),
            (By.XPATH, "//one-timeline-skill//div[contains(@class, 'drag-area')]"),
            (By.XPATH, "//svg/use[@xlink:href*='gripper']/../.."),
            (By.XPATH, "//svg/use[contains(@xlink:href, 'logo.svg')]/../.."),
            (By.XPATH, "//use[@xlink:href='bundles/gripper_grasp/logo.svg#icon']/../.."),
            (By.XPATH, "//use[@xlink:href*='gripper']/../.."),
        ]
        
        icon_element = self.selenium.try_multiple_locators(icon_locators, timeout=5)
        if icon_element:
            self.selenium.click_element_robust(icon_element)
            self.logger.info("✅ Clicked task icon - configuration dialog should open")
            return True
        else:
            self.logger.error("❌ Could not find task icon for configuration")
            return False

    def click_continue_button(self) -> bool:
        """Click the Continue button - WORKING VERSION, NO SLEEPS."""
        self.logger.info("➡️ Looking for Continue button...")
        
        try:
            # Wait for the button container to appear
            container_xpath = "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[4]/div[1]"
            
            container = self.wait_for_element((By.XPATH, container_xpath), timeout=10)
            if not container:
                self.logger.error("❌ Could not find button container")
                return False
            
            self.logger.info("✅ Found button container")
            
            # Find all buttons in this container
            buttons = container.find_elements(By.TAG_NAME, "button")
            self.logger.info(f"Found {len(buttons)} buttons in container")
            
            # Look for the Continue button specifically
            for i, button in enumerate(buttons):
                try:
                    button_text = button.text.strip().lower()
                    self.logger.info(f"Button {i+1}: '{button_text}' - Visible: {button.is_displayed()}, Enabled: {button.is_enabled()}")
                    
                    if "continue" in button_text:
                        self.logger.info(f"Found Continue button: '{button.text}'")
                        
                        if button.is_displayed() and button.is_enabled():
                            # Try direct click first
                            try:
                                button.click()
                                self.logger.info("✅ Clicked Continue button (direct click)")
                                return True
                            except Exception as e1:
                                self.logger.warning(f"Direct click failed: {e1}")
                                # Try JavaScript click
                                try:
                                    self.driver.execute_script("arguments[0].click();", button)
                                    self.logger.info("✅ Clicked Continue button (JavaScript click)")
                                    return True
                                except Exception as e2:
                                    self.logger.error(f"JavaScript click failed: {e2}")
                        else:
                            self.logger.warning(f"Continue button not clickable - Displayed: {button.is_displayed()}, Enabled: {button.is_enabled()}")
                
                except Exception as e:
                    self.logger.warning(f"Error checking button {i+1}: {e}")
                    continue
            
            self.logger.error("❌ No Continue button found in container")
            return False
                
        except Exception as e:
            self.logger.error(f"❌ Error clicking Continue button: {e}")
            return False

    def set_speed_value(self, speed: int) -> bool:
        """Set speed value using the correct editable div path."""
        self.logger.info(f"⚡ Setting speed to: {speed}")
        
        if not (10 <= speed <= 100):
            self.logger.error(f"❌ Speed {speed} out of range (10-100)")
            return False
        
        try:
            # Use the correct path with /div[1] at the end for the editable element
            speed_input_xpath = "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[4]/div/step/linear-slider/step/div/div[4]/div[1]"
            
            self.logger.info("⏳ Waiting for speed editable field to appear...")
            speed_element = self.wait_for_element((By.XPATH, speed_input_xpath), timeout=15)
            
            if not speed_element:
                self.logger.error("❌ Speed editable field not found")
                return False
            
            self.logger.info("✅ Found speed editable field")
            
            try:
                # Click to make it editable
                speed_element.click()
                self.logger.info("Clicked to make field editable")
                
                # Now just use simple key commands since cursor is in the field
                from selenium.webdriver.common.action_chains import ActionChains
                actions = ActionChains(self.driver)
                
                # Select all existing text and replace with new value
                actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)  # Ctrl+A to select all
                actions.send_keys(str(speed))  # Type new value
                actions.send_keys(Keys.ENTER)  # Press Enter
                actions.perform()
                
                self.logger.info(f"✅ Speed set to {speed} with simple key commands")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to set speed: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set speed: {e}")
            return False

    def set_force_value(self, force: int) -> bool:
        """Set grasping force value using the correct path."""
        self.logger.info(f"💪 Setting grasping force to: {force}")
        
        try:
            # Use the correct path for force field
            force_input_xpath = "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[7]/div/step/linear-slider/step/div/div[4]"
            
            self.logger.info("⏳ Waiting for force input field to appear...")
            force_element = self.wait_for_element((By.XPATH, force_input_xpath), timeout=15)
            
            if not force_element:
                self.logger.error("❌ Force input field not found")
                return False
            
            self.logger.info("✅ Found force input field")
            
            # Click to make it editable
            force_element.click()
            self.logger.info("Clicked to make force field editable")
            
            # Use same key commands as speed
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(self.driver)
            
            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)  # Ctrl+A
            actions.send_keys(str(force))  # Type new value
            actions.send_keys(Keys.ENTER)  # Press Enter
            actions.perform()
            
            self.logger.info(f"✅ Grasping force set to {force}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set grasping force: {e}")
            return False

    def set_load_value(self, load: int) -> bool:
        """Set load value using the correct path."""
        self.logger.info(f"⚖️ Setting load to: {load}")
        
        try:
            # Use the correct path for load field
            load_input_xpath = "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[10]/div/step/linear-slider/step/div/div[4]"
            
            self.logger.info("⏳ Waiting for load input field to appear...")
            load_element = self.wait_for_element((By.XPATH, load_input_xpath), timeout=15)
            
            if not load_element:
                self.logger.error("❌ Load input field not found")
                return False
            
            self.logger.info("✅ Found load input field")
            
            # Click to make it editable
            load_element.click()
            self.logger.info("Clicked to make load field editable")
            
            # Use same key commands as speed and force
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(self.driver)
            
            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)  # Ctrl+A
            actions.send_keys(str(load))  # Type new value
            actions.send_keys(Keys.ENTER)  # Press Enter
            actions.perform()
            
            self.logger.info(f"✅ Load set to {load}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set load: {e}")
            return False

    def wait_for_task_completion(self, timeout=30) -> bool:
        """Wait for current task to complete and robot to be truly ready - with error checking."""
        self.logger.info("⏳ Waiting for current task to complete...")
        
        try:
            # Wait for robot status to show Ready and no task is running
            for i in range(timeout * 2):  # Check every 0.5 seconds
                try:
                    # Check for task errors during wait (don't wait for them to appear)
                    if self.check_for_task_error(wait_for_error=False):
                        self.logger.warning("🚨 Task error detected during execution - task failed")
                        return True  # Return True because task "completed" (with error)
                    
                    # Check if robot shows Ready status
                    ready_element = self.driver.find_element(By.XPATH, "//*[contains(text(), 'Ready')]")
                    
                    # Check if execution button shows "run" (not "stop")
                    execution_button = self.selenium.try_multiple_locators([
                        (By.CSS_SELECTOR, "body > div:nth-child(2) > section > one-sidebar > div.sidebar-body > div > div.fixed-sections > footer > section > div > div"),
                        (By.XPATH, "/html/body/div[2]/section/one-sidebar/div[1]/div/div[2]/footer/section/div/div"),
                        (By.XPATH, "/html/body/div[2]/section/one-sidebar/div[2]/div/div[2]/footer/section/div/div"),
                    ], timeout=0.5)
                    
                    if execution_button:
                        # Check button state more thoroughly
                        button_classes = execution_button.get_attribute("class") or ""
                        button_text = execution_button.text.lower()
                        button_html = execution_button.get_attribute('outerHTML') or ""
                        
                        # If button shows stop indicators, task is still running
                        if ("stop" in button_classes or "stop" in button_text or 
                            "stop" in button_html.lower() or "cancel" in button_html.lower()):
                            if i % 10 == 0:  # Log every 5 seconds
                                self.logger.info(f"🔄 Task still running... (attempt {i+1}/{timeout*2})")
                            time.sleep(0.5)
                            continue
                        
                        # If button shows play/run indicators, task is complete
                        if ("play" in button_html.lower() or "run" in button_html.lower() or
                            not ("stop" in button_classes and "stop" in button_text)):
                            self.logger.info("✅ Task completed - Robot is ready for next task")
                            return True
                    
                except:
                    # If we can't find ready element or button, task might still be running
                    if i % 10 == 0:  # Log every 5 seconds
                        self.logger.info(f"🔄 Waiting for task completion... (attempt {i+1}/{timeout*2})")
                    pass
                
                time.sleep(0.5)
            
            self.logger.error(f"❌ Task did not complete within {timeout} seconds")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error waiting for task completion: {e}")
            return False

    def set_axis_offset(self, axis: str, value: float) -> bool:
        """Set offset value for X, Y, or Z axis."""
        self.logger.info(f"📐 Setting {axis.upper()} offset to {value}")
        
        try:
            # Button paths for X, Y, Z
            button_paths = {
                "x": "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/toggle-slider/step/div/div[1]/button[1]",
                "y": "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/toggle-slider/step/div/div[1]/button[2]",
                "z": "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/toggle-slider/step/div/div[1]/button[3]"
            }
            
            # Shared text field path
            text_field_path = "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/toggle-slider/step/div/div[2]/linear-slider/step/div/div[4]/div[1]"
            
            # 1. Click the axis button (X, Y, or Z)
            axis_button = self.wait_for_element((By.XPATH, button_paths[axis.lower()]), timeout=10)
            if not axis_button:
                self.logger.error(f"❌ {axis.upper()} button not found")
                return False
            
            axis_button.click()
            self.logger.info(f"✅ Clicked {axis.upper()} button")
            
            # 2. Set value in the shared text field
            text_field = self.wait_for_element((By.XPATH, text_field_path), timeout=10)
            if not text_field:
                self.logger.error(f"❌ Text field not found for {axis.upper()}")
                return False
            
            # Click to make editable and set value
            text_field.click()
            
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(self.driver)
            
            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)  # Ctrl+A
            actions.send_keys(str(value))  # Type value (supports negative numbers)
            actions.send_keys(Keys.ENTER)  # Press Enter
            actions.perform()
            
            self.logger.info(f"✅ {axis.upper()} offset set to {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set {axis.upper()} offset: {e}")
            return False

    def set_robot_speed(self, speed: int) -> bool:
        """Set robot movement speed."""
        self.logger.info(f"⚡ Setting robot speed to {speed}%")
        
        try:
            speed_field_path = "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[7]/div/step/linear-slider/step/div/div[4]/div[1]"
            
            speed_field = self.wait_for_element((By.XPATH, speed_field_path), timeout=10)
            if not speed_field:
                self.logger.error("❌ Robot speed field not found")
                return False
            
            speed_field.click()
            
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(self.driver)
            
            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)
            actions.send_keys(str(speed))
            actions.send_keys(Keys.ENTER)
            actions.perform()
            
            self.logger.info(f"✅ Robot speed set to {speed}%")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set robot speed: {e}")
            return False

    def set_robot_acceleration(self, acceleration: int) -> bool:
        """Set robot acceleration."""
        self.logger.info(f"🚀 Setting robot acceleration to {acceleration}%")
        
        try:
            accel_field_path = "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[10]/div/step/linear-slider/step/div/div[4]/div[1]"
            
            accel_field = self.wait_for_element((By.XPATH, accel_field_path), timeout=10)
            if not accel_field:
                self.logger.error("❌ Robot acceleration field not found")
                return False
            
            accel_field.click()
            
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(self.driver)
            
            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)
            actions.send_keys(str(acceleration))
            actions.send_keys(Keys.ENTER)
            actions.perform()
            
            self.logger.info(f"✅ Robot acceleration set to {acceleration}%")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set robot acceleration: {e}")
            return False

    def check_for_task_error(self, wait_for_error: bool = True) -> bool:
        """Check if there's a task error displayed and log it. Returns True if error found."""
        try:
            if wait_for_error:
                # Wait a bit for error messages to appear after task completion
                import time
                time.sleep(2)
            
            # Check for error bar
            error_bar = self.selenium.try_multiple_locators([
                (By.XPATH, "/html/body/div[2]/section/section/one-error-bar/div"),
                (By.CSS_SELECTOR, "one-error-bar div"),
                (By.CSS_SELECTOR, ".error-window"),
            ], timeout=1)
            
            if not error_bar:
                return False  # No error bar found
            
            # Wait a bit more for the error message text to load
            import time
            time.sleep(1)
            
            # Look for error message within the error bar
            error_message_locators = [
                (By.XPATH, ".//div[@data-bind='text: errorMessage']"),
                (By.XPATH, ".//*[contains(@data-bind, 'errorMessage')]"),
                (By.XPATH, ".//div[contains(text(), 'Could not')]"),
                (By.XPATH, ".//div[contains(text(), 'Take:')]"),
                (By.XPATH, ".//div[contains(text(), 'timeout')]"),
            ]
            
            for locator in error_message_locators:
                try:
                    error_element = error_bar.find_element(*locator)
                    if error_element and error_element.text.strip():
                        error_text = error_element.text.strip()
                        if error_text and len(error_text) > 5:  # Make sure it's a real message
                            self.logger.error(f"🚨 TASK ERROR: {error_text}")
                            return True
                except:
                    continue
            
            # Try to get any text from the error bar as fallback
            try:
                error_bar_text = error_bar.text.strip()
                if error_bar_text and len(error_bar_text) > 5:
                    self.logger.error(f"🚨 TASK ERROR: {error_bar_text}")
                    return True
            except:
                pass
            
            # If error bar exists but still no readable message after waiting, it might not be an error
            self.logger.debug("Error bar detected but no readable error message found")
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking task error: {e}")
            return False

    def activate_field_step(self, step_number: int) -> bool:
        """Activate a field step by clicking on its container (for fields 2 and 3)."""
        step_paths = {
            2: "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/div/step[2]",
            3: "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/div/step[3]"
        }
        
        if step_number not in step_paths:
            return True  # Step 1 is already active
        
        try:
            step_container = self.wait_for_element((By.XPATH, step_paths[step_number]), timeout=5)
            if not step_container:
                self.logger.error(f"❌ Step {step_number} container not found")
                return False
            
            # Click to activate the step
            if not self.selenium.click_element_robust(step_container):
                self.logger.error(f"❌ Failed to activate step {step_number}")
                return False
            
            self.logger.info(f"✅ Activated step {step_number}")
            import time
            time.sleep(0.1)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to activate step {step_number}: {e}")
            return False

    def set_field_value(self, field_path: str, value, field_name: str, step_number: int = 1) -> bool:
        """Generic method to set any field value - handles field activation."""
        self.logger.info(f"📝 Setting {field_name} to: {value}")
        
        try:
            # For steps 2 and 3, we need to activate the step first
            if step_number > 1:
                if not self.activate_field_step(step_number):
                    return False
            
            # Find the text field
            field_element = self.wait_for_element((By.XPATH, field_path), timeout=10)
            if not field_element:
                self.logger.error(f"❌ {field_name} field not found")
                return False
            
            # Click the text field to focus it
            if not self.selenium.click_element_robust(field_element):
                self.logger.error(f"❌ Failed to click {field_name} field")
                return False
            
            self.logger.info(f"✅ Successfully clicked {field_name} field")
            
            # Give the field a moment to become active
            import time
            time.sleep(0.1)
            
            # Use ActionChains to edit the field
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(self.driver)
            
            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL)  # Select all
            actions.send_keys(str(value))  # Type new value
            actions.send_keys(Keys.ENTER)  # Confirm
            actions.perform()
            
            self.logger.info(f"✅ {field_name} set to {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set {field_name}: {e}")
            return False

    def click_stop_button(self) -> bool:
        """Click the stop/cancel task button when task fails."""
        self.logger.info("🛑 Clicking stop button to cancel failed task...")
        
        stop_button_locators = [
            (By.XPATH, "/html/body/div[2]/section/one-sidebar/div[2]/div/div[2]/footer/section/div/div"),
            (By.XPATH, "/html/body/div[2]/section/one-sidebar/div[1]/div/div[2]/footer/section/div/div"),  # Alternative path
            (By.CSS_SELECTOR, "one-sidebar footer section div div"),
            (By.XPATH, "//footer//section//div//div[contains(@class, 'execution-button')]"),
        ]
        
        stop_button = self.selenium.try_multiple_locators(stop_button_locators, timeout=5)
        if stop_button:
            if self.selenium.click_element_robust(stop_button):
                self.logger.info("✅ Stop button clicked - task cancelled")
                return True
            else:
                self.logger.error("❌ Failed to click stop button")
                return False
        else:
            self.logger.warning("⚠️ Stop button not found")
            return False

    # ========== HIGH-LEVEL GRIPPER COMMANDS ==========

    def configure_gripper_open(self, speed: int = 20) -> bool:
        """Configure Gripper_open task with specified speed - NO SLEEPS."""
        self.logger.info(f"⚙️ Configuring Gripper_open with speed={speed}")
        
        try:
            # 0. Wait for any current task to complete first
            if not self.wait_for_task_completion(timeout=10):
                self.logger.warning("⚠️ Previous task still running, waiting...")
            
            # 1. Select Gripper_open task
            if not self.select_task_from_list("Gripper_open"):
                return False
            
            # 2. Click task icon to configure
            if not self.click_task_icon_for_config():
                return False
            
            # 3. Click Continue button (width -> speed tab)
            if not self.click_continue_button():
                self.logger.error("❌ Failed to click Continue button")
                return False
            
            # 4. Set speed value
            if not self.set_speed_value(speed):
                self.logger.error("❌ Failed to set speed value")
                return False
            
            # 5. Click Continue button again (close dialog)
            if not self.click_continue_button():
                self.logger.error("❌ Failed to click Continue button to close")
                return False
            
            self.logger.info(f"✅ Gripper_open configured (speed={speed})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to configure gripper open: {e}")
            return False

    def configure_gripper_close(self, speed: int = 50, force: int = 80, load: int = 400) -> bool:
        """Configure Gripper_close task with speed, force, and load parameters."""
        self.logger.info(f"⚙️ Configuring Gripper_close with speed={speed}, force={force}, load={load}")
        
        # Validate parameters
        if not (10 <= speed <= 100):
            self.logger.error(f"❌ Speed {speed} out of range (10-100)")
            return False
        if not (20 <= force <= 100):
            self.logger.error(f"❌ Force {force} out of range (20-100)")
            return False
        if not (10 <= load <= 1000):
            self.logger.error(f"❌ Load {load} out of range (10-1000)")
            return False
        
        try:
            # 0. Wait for any current task to complete first
            if not self.wait_for_task_completion(timeout=10):
                self.logger.warning("⚠️ Previous task still running, waiting...")
            
            # 1. Select Gripper_close task
            if not self.select_task_from_list("Gripper_close"):
                return False
            
            # 2. Click task icon to configure
            if not self.click_task_icon_for_config():
                return False
            
            # 3. Skip Gripper width tab -> Continue
            self.logger.info("📏 Skipping gripper width configuration...")
            if not self.click_continue_button():
                self.logger.error("❌ Failed to skip gripper width")
                return False
            
            # 4. Configure Gripper speed -> Continue
            self.logger.info(f"⚡ Setting gripper speed to {speed}...")
            if not self.set_speed_value(speed):
                self.logger.error("❌ Failed to set gripper speed")
                return False
            
            if not self.click_continue_button():
                self.logger.error("❌ Failed to continue from speed tab")
                return False
            
            # 5. Configure Grasping force -> Continue
            self.logger.info(f"💪 Setting grasping force to {force}...")
            if not self.set_force_value(force):
                self.logger.error("❌ Failed to set grasping force")
                return False
            
            if not self.click_continue_button():
                self.logger.error("❌ Failed to continue from force tab")
                return False
            
            # 6. Configure Load -> Continue (to close)
            self.logger.info(f"⚖️ Setting load to {load}...")
            if not self.set_load_value(load):
                self.logger.error("❌ Failed to set load")
                return False
            
            if not self.click_continue_button():
                self.logger.error("❌ Failed to close dialog from load tab")
                return False
            
            self.logger.info(f"✅ Gripper_close configured (speed={speed}, force={force}, load={load})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to configure gripper close: {e}")
            return False


    def gripper_open(self) -> bool:
        """Execute gripper open command."""
        self.logger.info("🔓 Opening gripper...")
        
        if not self.select_task_from_list("Gripper_open"):
            return False
        
        if not self.click_execution_button():
            return False
        
        if not self.click_confirm_button():
            return False
        
        # Wait for task to actually complete (not just show Ready)
        if not self.wait_for_task_completion():
            return False
        
        self.logger.info("✅ Gripper opened successfully")
        return True

    def gripper_close(self) -> bool:
        """Execute gripper close command."""
        self.logger.info("🤏 Closing gripper...")
        
        # First, make sure no task is currently running
        if not self.wait_for_task_completion(timeout=10):
            self.logger.warning("⚠️ Previous task still running, waiting...")
        
        if not self.select_task_from_list("Gripper_close"):
            return False
        
        if not self.click_execution_button():
            return False
        
        if not self.click_confirm_button():
            return False
        
        # Wait for task to complete
        if not self.wait_for_task_completion():
            return False
        
        self.logger.info("✅ Gripper closed successfully")
        return True

    def move_robot(self, x: float = 0, y: float = 0, z: float = 0, speed: int = 5, acceleration: int = 5) -> bool:
        """Move robot with relative motion and execute immediately."""
        self.logger.info(f"🤖 Moving robot: X={x}, Y={y}, Z={z}, Speed={speed}%, Accel={acceleration}%")
        
        # Validate parameters
        if not (5 <= speed <= 100):
            self.logger.error(f"❌ Speed {speed} out of range (5-100%)")
            return False
        if not (5 <= acceleration <= 100):
            self.logger.error(f"❌ Acceleration {acceleration} out of range (5-100%)")
            return False
        
        try:
            # 0. Wait for any current task to complete
            if not self.wait_for_task_completion(timeout=10):
                self.logger.warning("⚠️ Previous task still running, waiting...")
            
            # 1. Select Move_robot task
            if not self.select_task_from_list("Move_robot"):
                return False
            
            # 2. Click task icon to configure
            if not self.click_task_icon_for_config():
                return False
            
            # 3. Set ALL X, Y, Z offsets explicitly (including 0 values to clear previous configs)
            if not self.set_axis_offset("x", x):
                return False
            
            if not self.set_axis_offset("y", y):
                return False
            
            if not self.set_axis_offset("z", z):
                return False
            
            # 4. Continue from OFFSET to FRAME tab
            self.logger.info("📐 Continuing from OFFSET to FRAME...")
            if not self.click_continue_button():
                self.logger.error("❌ Failed to continue from OFFSET to FRAME")
                return False
            
            # 5. Continue from FRAME to SPEED tab (skip frame selection)
            self.logger.info("🖼️ Continuing from FRAME to SPEED...")
            if not self.click_continue_button():
                self.logger.error("❌ Failed to continue from FRAME to SPEED")
                return False
            
            # 6. Set speed in SPEED tab
            self.logger.info(f"⚡ Setting robot speed to {speed}%...")
            if not self.set_robot_speed(speed):
                self.logger.error("❌ Failed to set robot speed")
                return False
            
            # 7. Continue from SPEED to ACCELERATION tab
            if not self.click_continue_button():
                self.logger.error("❌ Failed to continue from SPEED to ACCELERATION")
                return False
            
            # 8. Set acceleration in ACCELERATION tab
            self.logger.info(f"🚀 Setting robot acceleration to {acceleration}%...")
            if not self.set_robot_acceleration(acceleration):
                self.logger.error("❌ Failed to set robot acceleration")
                return False
            
            # 9. Continue to close dialog
            if not self.click_continue_button():
                self.logger.error("❌ Failed to close configuration dialog")
                return False
            
            # 10. Execute the task immediately
            self.logger.info("▶️ Executing robot movement...")
            if not self.click_execution_button():
                return False
            
            if not self.click_confirm_button():
                return False
            
            if not self.wait_for_task_completion():
                return False
            
            self.logger.info(f"✅ Robot moved successfully: X={x}, Y={y}, Z={z}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to move robot: {e}")
            return False

    def suction_on(self, load: int = None, vacuum: int = None, timeout: float = None) -> bool:
        """Execute suction_on command. If parameters provided, configure first."""
        
        # If any parameters provided, configure first
        if load is not None or vacuum is not None or timeout is not None:
            # Use default values for unspecified parameters
            load = load if load is not None else 100
            vacuum = vacuum if vacuum is not None else 650
            timeout = timeout if timeout is not None else 5.0
            
            # Validate parameters
            if not (0 <= load <= 2000):
                self.logger.error(f"❌ Load {load} out of range (0-2000)")
                return False
            if not (550 <= vacuum <= 750):
                self.logger.error(f"❌ Vacuum {vacuum} out of range (550-750)")
                return False
            if not (0.5 <= timeout <= 10):
                self.logger.error(f"❌ Timeout {timeout} out of range (0.5-10)")
                return False
            
            self.logger.info(f"⚙️ Configuring Suction_on with load={load}, vacuum={vacuum}, timeout={timeout}")
            
            try:
                # Wait for any current task to complete first
                if not self.wait_for_task_completion(timeout=10):
                    self.logger.warning("⚠️ Previous task still running, waiting...")
                
                # Select Suction_on task
                if not self.select_task_from_list("Suction_on"):
                    return False
                
                # Click task icon to configure
                if not self.click_task_icon_for_config():
                    return False
                
                # Set all three values - note the step numbers!
                self.logger.info("📝 Setting suction parameters...")
                
                # Step 1 (already active)
                if not self.set_field_value(
                    "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/div/step[1]/linear-slider/step/div/div[4]/div[1]",
                    load, "suction load", step_number=1
                ):
                    return False
                
                # Step 2 (needs activation)
                if not self.set_field_value(
                    "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/div/step[2]/linear-slider/step/div/div[4]/div[1]",
                    vacuum, "suction vacuum", step_number=2
                ):
                    return False
                
                # Step 3 (needs activation)
                if not self.set_field_value(
                    "/html/body/div[2]/section/section/section/one-timeline/div[3]/div/one-container/div/one-timeline-skill/div/one-context-menu/div/div[3]/div[1]/div/step/div/step[3]/linear-slider/step/div/div[4]/div[1]",
                    timeout, "suction timeout", step_number=3
                ):
                    return False
                
                self.logger.info("✅ All suction parameters configured")
                
                # Click Continue to close dialog
                if not self.click_continue_button():
                    self.logger.error("❌ Failed to close configuration dialog")
                    return False
                
                self.logger.info(f"✅ Suction_on configured (load={load}, vacuum={vacuum}, timeout={timeout})")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to configure suction_on: {e}")
                return False
        else:
            self.logger.info("🌪️ Executing suction_on with current configuration...")
            
            # Wait for any current task to complete
            if not self.wait_for_task_completion(timeout=10):
                self.logger.warning("⚠️ Previous task still running, waiting...")
            
            # Just select the task
            if not self.select_task_from_list("Suction_on"):
                return False
        
        # Execute the task (same for both configured and non-configured)
        if not self.click_execution_button():
                return False
        
        if not self.click_confirm_button():
            return False
        
        # Calculate appropriate timeout based on configured timeout
        if load is not None or vacuum is not None or timeout is not None:
            # If we configured it, use the timeout + buffer
            wait_timeout = int((timeout if timeout is not None else 5.0) + 15)  # Add 15 second buffer
        else:
            # Use default timeout
            wait_timeout = 30
        
        # Wait for task completion (now checks for errors during wait)
        task_completed = self.wait_for_task_completion(timeout=wait_timeout)
        
        # Final error check (in case error appeared after completion)
        self.logger.info("🔍 Final check for task errors...")
        if self.check_for_task_error(wait_for_error=True):
            # Stop the failed task
            self.click_stop_button()
            self.logger.error("❌ Suction_on failed with error")
            return False
        
        if not task_completed:
            self.logger.error("❌ Suction_on did not complete properly")
            # Try to stop if it's still running
            self.click_stop_button()
            return False
        
        self.logger.info("✅ Suction_on executed successfully")
        return True