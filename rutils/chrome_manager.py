"""Chrome WebDriver management and configuration."""

import os
import tempfile
import shutil
import subprocess
from pathlib import Path
import logging
import time
from typing import Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager as WebDriverManagerCDM
from webdriver_manager.core.os_manager import ChromeType
from .config import Config


class ChromeDriverManager:
    """Manages Chrome WebDriver setup and configuration."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def create_driver(self, headless: bool = False) -> webdriver.Chrome:
        """Create Chrome WebDriver with automatic driver management."""
        # Use webdriver-manager to automatically get compatible ChromeDriver
        chromedriver_path = self._get_chromedriver_auto()
        
        self.logger.info(f"🚗 Using ChromeDriver: {chromedriver_path}")
        service = ChromeService(executable_path=chromedriver_path)
        
        # Use FIXED persistent profile directory
        persistent_profile = Path.home() / "franka_robot_profile"
        
        # Clean up any existing locks but keep the profile
        if persistent_profile.exists():
            self._cleanup_profile_locks(persistent_profile)
        
        try:
            return self._create_driver_with_profile(service, persistent_profile, headless, persistent=True)
        except Exception as e:
            self.logger.warning(f"⚠️ Persistent profile failed: {e}")
            # Remove the problematic profile and try again
            if persistent_profile.exists():
                try:
                    shutil.rmtree(persistent_profile)
                    self.logger.info("🗑️ Removed problematic persistent profile")
                except:
                    pass
            
            self.logger.info("🔄 Using temporary profile...")
            temp_profile = Path(tempfile.mkdtemp(prefix="franka_chrome_"))
            return self._create_driver_with_profile(service, temp_profile, headless, persistent=False)
    
    def _get_chromedriver_auto(self) -> str:
        """Automatically get compatible ChromeDriver using webdriver-manager."""
        try:
            self.logger.info("📥 Auto-downloading compatible ChromeDriver...")
            
            # Try Chromium first, then fallback to Chrome
            try:
                self.logger.info("🔍 Trying Chromium...")
                chromedriver_path = WebDriverManagerCDM(chrome_type=ChromeType.CHROMIUM).install()
                self.logger.info("✅ ChromeDriver for Chromium auto-download successful")
                return chromedriver_path
            except Exception as chromium_error:
                self.logger.info(f"Chromium failed: {chromium_error}, trying Chrome...")
                chromedriver_path = WebDriverManagerCDM().install()
                self.logger.info("✅ ChromeDriver for Chrome auto-download successful")
                return chromedriver_path
                
        except Exception as e:
            self.logger.error(f"❌ Auto-download failed: {e}")
            # Fallback to system driver
            system_driver = self._find_system_chromedriver()
            if system_driver:
                self.logger.warning("⚠️ Using system ChromeDriver as fallback")
                return system_driver
            raise RuntimeError("No ChromeDriver available")
    
    def _find_system_chromedriver(self) -> Optional[str]:
        """Find system-installed ChromeDriver."""
        locations = [
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver", 
            "/opt/google/chrome/chromedriver",
            "/snap/bin/chromium.chromedriver",
            "/usr/bin/chromium-driver",
        ]
        
        # Check PATH
        path_location = shutil.which("chromedriver")
        if path_location:
            locations.insert(0, path_location)
        
        for location in locations:
            if location and os.path.isfile(location) and os.access(location, os.X_OK):
                return location
        return None
    
    def _create_driver_with_profile(self, service, profile_path: Path, headless: bool, persistent: bool) -> webdriver.Chrome:
        """Create driver with specified profile."""
        if persistent:
            profile_path.mkdir(parents=True, exist_ok=True)
            os.chmod(profile_path, 0o755)
            self._cleanup_profile_locks(profile_path)
            self.logger.info(f"📂 Using persistent profile: {profile_path}")
        else:
            self.logger.info(f"📂 Using temporary profile: {profile_path}")
        
        options = ChromeOptions()
        
        # Try to use Chromium binary if available
        chromium_paths = [
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/snap/bin/chromium",
            "/usr/bin/google-chrome",  # Fallback to Chrome
        ]
        
        for chromium_path in chromium_paths:
            if os.path.isfile(chromium_path):
                options.binary_location = chromium_path
                self.logger.info(f"🌐 Using browser: {chromium_path}")
                break
        
        options.add_argument(f"--user-data-dir={profile_path}")
        
        # Essential options for stability
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-background-timer-throttling")
        
        # Security for robot interface
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--allow-running-insecure-content")
        options.add_argument("--disable-web-security")
        options.add_argument("--allow-insecure-localhost")
        
        # Performance and compatibility
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        
        # Browser stability options
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-features=TranslateUI")
        options.add_argument("--disable-ipc-flooding-protection")
        
        if headless:
            options.add_argument("--headless=new")
        
        # Suppress logs and detection
        options.add_argument("--log-level=3")
        options.add_argument("--silent")
        options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(1)
        driver.set_page_load_timeout(30)
        
        self.logger.info("✅ Chrome WebDriver created successfully")
        return driver
    
    def _cleanup_profile_locks(self, profile_path: Path) -> None:
        """Remove Chrome lock files."""
        lock_files = ["SingletonLock", "SingletonSocket", "SingletonCookie"]
        for lock_file in lock_files:
            lock_path = profile_path / lock_file
            if lock_path.exists():
                try:
                    lock_path.unlink()
                    self.logger.debug(f"Removed lock file: {lock_file}")
                except Exception as e:
                    self.logger.debug(f"Could not remove {lock_file}: {e}")
    
    def cleanup_all_chrome_processes(self) -> None:
        """Clean up Chrome processes."""
        self.logger.info("🧹 Cleaning up all Chrome processes...")
        
        commands = [
            ["pkill", "-f", "chrome"],
            ["pkill", "-f", "chromium"],
            ["pkill", "-f", "chromedriver"],
            ["killall", "chrome"],
            ["killall", "chromium"],
            ["killall", "chromedriver"],
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, capture_output=True, timeout=3)
            except:
                pass
        
        # Clean temp directories
        import glob
        temp_dirs = glob.glob("/tmp/franka_chrome_*")
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass