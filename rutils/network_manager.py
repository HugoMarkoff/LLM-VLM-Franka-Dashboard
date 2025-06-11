"""Network configuration management for robot communication."""

import subprocess
import logging
import re
from typing import Optional
from .config import Config


class NetworkManager:
    """Handles network configuration for robot communication."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def check_network_configured(self) -> bool:
        """Check if the required network configuration exists."""
        try:
            result = subprocess.run(
                ["ip", "addr", "show", self.config.network_interface],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return self.config.local_ip in result.stdout
            return False
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            self.logger.warning(f"Failed to check network configuration: {e}")
            return False
    
    def check_interface_exists(self) -> bool:
        """Check if the specified interface exists."""
        try:
            result = subprocess.run(
                ["ip", "link", "show", self.config.network_interface],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def detect_usb_ethernet_interface(self) -> Optional[str]:
        """Auto-detect USB Ethernet interface."""
        try:
            # Get all network interfaces
            result = subprocess.run(["ip", "link", "show"], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Look for USB Ethernet interfaces (usually start with enx)
                usb_interfaces = re.findall(r'\d+: (enx[a-f0-9]{12})', result.stdout)
                
                if usb_interfaces:
                    interface = usb_interfaces[0]
                    self.logger.info(f"🔍 Auto-detected USB Ethernet interface: {interface}")
                    return interface
                    
            return None
        except Exception as e:
            self.logger.warning(f"Failed to auto-detect interface: {e}")
            return None
    
    def configure_network(self) -> bool:
        """Configure network interface for robot communication."""
        # Try to auto-detect if interface not found
        if not self.check_interface_exists():
            self.logger.warning(f"⚠️ Interface {self.config.network_interface} not found, trying auto-detection...")
            detected = self.detect_usb_ethernet_interface()
            if detected:
                self.logger.info(f"✅ Switching to detected interface: {detected}")
                self.config.network_interface = detected
            else:
                self.logger.error(f"❌ Interface {self.config.network_interface} not found and auto-detection failed")
                return False
        
        if self.check_network_configured():
            self.logger.info(f"✅ Network already configured: {self.config.network_assignment}")
            return True
        
        self.logger.info(f"🌐 Configuring network: {self.config.network_assignment} on {self.config.network_interface}")
        
        try:
            # Add IP address to interface
            cmd = [
                "sudo", "ip", "addr", "add",
                self.config.network_assignment,
                "dev", self.config.network_interface
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.logger.info("✅ Network configuration successful")
                return True
            else:
                self.logger.error(f"❌ Network configuration failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Network configuration timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Network configuration error: {e}")
            return False
    
    def test_robot_connectivity(self) -> bool:
        """Test if robot is reachable."""
        self.logger.info(f"🔍 Testing connectivity to {self.config.robot_ip}")
        
        try:
            result = subprocess.run(
                ["ping", "-c", "2", "-W", "3", self.config.robot_ip],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.info("✅ Robot is reachable")
                return True
            else:
                self.logger.warning("⚠️ Robot is not reachable via ping")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning("⚠️ Ping test timed out")
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ Connectivity test failed: {e}")
            return False