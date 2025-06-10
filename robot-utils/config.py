"""Configuration management for Franka automation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Application configuration."""
    
    # Network settings
    robot_ip: str = "172.16.0.2"
    local_ip: str = "172.16.0.1"
    network_interface: str = "enp2s0"
    subnet: str = "24"
    
    # Robot authentication
    username: str = "Panda"
    password: str = "panda1234"
    
    # Timeouts (seconds)
    short_timeout: int = 3
    default_timeout: int = 10
    long_timeout: int = 10
    
    # Chrome profile settings
    franka_profile_name: str = "FrankaRobot"
    
    @property
    def robot_url(self) -> str:
        return f"https://{self.robot_ip}/desk/"
    
    @property
    def network_assignment(self) -> str:
        return f"{self.local_ip}/{self.subnet}"
    
    @property
    def franka_profile_path(self) -> str:
        """Get path to persistent Franka Chrome profile."""
        home_dir = Path.home()
        profile_base = home_dir / ".config" / "franka-automation"
        return str(profile_base / "chrome-profile")
    
    @property
    def chromedriver_path(self) -> str:
        """Get path for custom ChromeDriver installation."""
        home_dir = Path.home()
        driver_base = home_dir / ".config" / "franka-automation"
        return str(driver_base / "chromedriver")