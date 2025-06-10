"""Element locators for Franka robot interface - FIXED SELECTORS."""

from selenium.webdriver.common.by import By
from typing import List, Tuple


class FrankaLocators:
    """Centralized locators for Franka robot interface elements."""
    
    # Authentication locators
    LOGIN_USERNAME = [
        (By.CSS_SELECTOR, "input[type='text']"),
        (By.CSS_SELECTOR, "input[name='username']"),
        (By.CSS_SELECTOR, "[name*='user']"),
    ]
    
    LOGIN_PASSWORD = [
        (By.CSS_SELECTOR, "input[type='password']"),
        (By.CSS_SELECTOR, "[name*='password']"),
    ]
    
    LOGIN_SUBMIT = [
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.CSS_SELECTOR, "input[type='submit']"),
        (By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign')]"),
    ]
    
    # Control request locators - SAFER SELECTORS ONLY
    REQUEST_CONTROL_BUTTON = [
        (By.XPATH, "//button[text()='REQUEST CONTROL']"),  # Exact match only
        (By.XPATH, "//button[contains(text(), 'REQUEST CONTROL')]"),
        (By.XPATH, "//button[normalize-space()='REQUEST CONTROL']"),
        # Remove generic button selectors that might click wrong buttons
    ]

    CONTROL_DIALOG = [
        (By.XPATH, "//*[contains(text(), 'has control')]"),
        (By.XPATH, "//*[contains(text(), 'control')]"),  # More generic
        (By.CSS_SELECTOR, ".modal"),
        (By.CSS_SELECTOR, ".dialog"),
        (By.CSS_SELECTOR, "[class*='modal']"),
        (By.CSS_SELECTOR, "[class*='dialog']"),
        (By.XPATH, "//div[contains(@style, 'z-index')]"),  # Modal usually has z-index
    ]
    
    # Robot status locators
    READY_STATUS = [
        (By.XPATH, "//*[contains(text(), 'Ready')]"),
        (By.CSS_SELECTOR, "[class*='ready']"),
    ]
    
    JOINTS_LOCKED = [
        (By.XPATH, "//*[contains(text(), 'Joints locked')]"),
        (By.XPATH, "//*[contains(text(), 'locked')]"),
    ]
    
    FREE_MOVES = [
        (By.XPATH, "//div[@title='Free Moves']"),
        (By.XPATH, "//*[contains(text(), 'FREE MOVES')]"),
        (By.CSS_SELECTOR, "[title*='Free Moves']"),
    ]
    
    UNLOCK_BUTTON = [
        (By.XPATH, "//button[contains(text(), 'Open')]"),
        (By.XPATH, "//button[contains(text(), 'Unlock')]"),
    ]
    
    # Task locators
    @staticmethod
    def task_selector(name: str) -> List[Tuple[By, str]]:
        """Get locators for a specific task name."""
        return [
            (By.XPATH, f"//span[@title='{name}']"),
            (By.XPATH, f"//*[contains(text(), '{name}')]"),
        ]
    
    SKILL_ICON = [
        (By.CSS_SELECTOR, ".skill-icon"),
        (By.CSS_SELECTOR, "[class*='skill']"),
    ]
    
    # Wizard controls
    WIZARD_CLOSE = [
        (By.CSS_SELECTOR, ".close"),
        (By.CSS_SELECTOR, "[data-testid='close']"),
        (By.XPATH, "//button[contains(@class, 'close')]"),
    ]
    
    WIZARD_NEXT = [
        (By.XPATH, "//button[contains(@class, 'check')]"),
        (By.XPATH, "//button[contains(text(), 'Continue')]"),
        (By.XPATH, "//button[contains(text(), 'Next')]"),
    ]
    
    CIRCLE_BUTTON = [
        (By.XPATH, "//button[contains(@class, 'circle')]"),
    ]
    
    # Sliders
    SLIDER_DISPLAY = [
        (By.CSS_SELECTOR, ".slider-value-display"),
        (By.CSS_SELECTOR, "[class*='slider-value']"),
    ]
    
    SLIDER_INPUT = [
        (By.CSS_SELECTOR, ".slider-value-wrapper input"),
        (By.CSS_SELECTOR, "input[type='number']"),
    ]
    
    # Execution
    RUN_BUTTON = [
        (By.CSS_SELECTOR, ".execution-button"),
        (By.XPATH, "//button[contains(text(), 'Run')]"),
    ]
    
    CONFIRM_BUTTON = [
        (By.XPATH, "//button[contains(text(), 'Confirm')]"),
        (By.XPATH, "//button[contains(text(), 'OK')]"),
    ]