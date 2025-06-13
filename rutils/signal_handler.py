"""Signal handling utilities that work in both main and background threads."""

import signal
import threading
import logging
from typing import Optional


class GracefulKiller:
    """Signal handler that works in both main thread and background threads."""
    
    def __init__(self, automation_instance=None):
        self.automation = automation_instance
        self.kill_now = False
        self._original_handlers = {}
        self.logger = logging.getLogger(__name__)
        
        # Only set up signal handlers if we're in the main thread
        if threading.current_thread() is threading.main_thread():
            self._setup_signal_handlers()
            self.logger.info("✅ Signal handlers registered (main thread)")
        else:
            self.logger.info("⚠️ Skipping signal handlers (background thread)")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers - only works in main thread."""
        try:
            # Store original handlers
            self._original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)
            self._original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception as e:
            self.logger.warning(f"Could not set up signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"🛑 Received signal {signum}")
        self.kill_now = True
        
        if self.automation:
            try:
                self.automation.cleanup()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
    
    def restore_handlers(self):
        """Restore original signal handlers."""
        if threading.current_thread() is threading.main_thread():
            for sig, handler in self._original_handlers.items():
                try:
                    signal.signal(sig, handler)
                except Exception as e:
                    self.logger.warning(f"Could not restore handler for signal {sig}: {e}")