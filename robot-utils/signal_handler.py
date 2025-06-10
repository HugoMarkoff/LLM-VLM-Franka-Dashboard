"""Signal handling for graceful shutdown."""

import signal
import sys


class GracefulKiller:
    """Handle graceful shutdown on signals."""
    
    def __init__(self, automation_instance=None):
        self.kill_now = False
        self.automation = automation_instance
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        """Handle graceful shutdown."""
        print("\n🛑 Graceful shutdown initiated...")
        self.kill_now = True
        if self.automation:
            self.automation.cleanup()
        sys.exit(0)