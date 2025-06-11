"""Enhanced Selenium automation utilities."""

import time
import logging
from typing import Optional, List, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import (
    TimeoutException,
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
)
from .config import Config


class SeleniumHelper:
    """Enhanced Selenium automation utilities."""
    
    def __init__(self, driver: webdriver.Chrome, config: Config, logger: logging.Logger):
        self.driver = driver
        self.config = config
        self.logger = logger
    
    def wait_for_element(
        self,
        locator: Tuple[By, str],
        timeout: int = None,
        condition = EC.presence_of_element_located,
        save_debug_on_timeout: bool = True  # NEW PARAMETER
    ) -> WebElement:
        """Wait for element with specified condition."""
        timeout = timeout or self.config.default_timeout
        try:
            return WebDriverWait(self.driver, timeout).until(condition(locator))
        except TimeoutException:
            # Only save debug info if explicitly requested
            if save_debug_on_timeout:
                self.save_debug_info(f"timeout_waiting_for_{locator[1].replace('/', '_')}")
            raise
        
    def find_element_safe(self, locator: Tuple[By, str]) -> Optional[WebElement]:
        """Find element without throwing exception."""
        try:
            return self.driver.find_element(*locator)
        except NoSuchElementException:
            return None
    
    def click_element_robust(self, element: WebElement, max_attempts: int = 3) -> bool:
        """Click element with comprehensive retry logic."""
        for attempt in range(max_attempts):
            try:
                # Ensure element is in view
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center', behavior: 'auto'});",
                    element
                )
                time.sleep(0.2)
                
                # Try normal click
                try:
                    ActionChains(self.driver).move_to_element(element).pause(0.1).click().perform()
                    return True
                except ElementClickInterceptedException:
                    # Fallback to JavaScript click
                    self.driver.execute_script("arguments[0].click();", element)
                    return True
                    
            except StaleElementReferenceException:
                self.logger.debug(f"Stale element on attempt {attempt + 1}")
                time.sleep(0.5)
                continue
            except Exception as e:
                self.logger.debug(f"Click attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.5)
        
        return False
    
    def is_browser_alive(self) -> bool:
        """Check if browser session is still alive."""
        try:
            # Simple check that doesn't cause issues
            _ = self.driver.current_url
            return True
        except:
            return False

    def try_multiple_locators(self, locators: List[Tuple[By, str]], timeout: int = None) -> Optional[WebElement]:
        """Try multiple locators and return first found element - NO DEBUG SPAM."""
        timeout = timeout or self.config.short_timeout
        
        for locator in locators:
            try:
                if not self.is_browser_alive():
                    self.logger.error("❌ Browser session died")
                    return None
                
                # Don't save debug files for existence checks!
                return self.wait_for_element(locator, timeout, save_debug_on_timeout=False)
            except TimeoutException:
                continue  # Expected - just try next locator
            except Exception as e:
                self.logger.warning(f"Locator {locator} failed: {e}")
                if "invalid session" in str(e) or "disconnected" in str(e):
                    self.logger.error("❌ Browser crashed during element search")
                    return None
                continue
        return None
    
    def save_debug_info(self, suffix: str = "") -> None:
        """Save debug information."""
        try:
            timestamp = int(time.time())
            screenshot_path = f"franka_debug_{timestamp}_{suffix}.png"
            html_path = f"franka_debug_{timestamp}_{suffix}.html"
            
            self.driver.save_screenshot(screenshot_path)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            
            self.logger.info(f"💾 Debug files saved: {screenshot_path}, {html_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save debug info: {e}")