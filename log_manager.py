#!/usr/bin/env python3
"""
Log Manager for comprehensive system logging with automatic rotation and cleanup.
Captures all output from run_system script with 5-day retention policy.
"""

import os
import sys
import logging
import glob
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import threading
import time

class SystemLogManager:
    """Manages comprehensive logging for the run_system script with automatic cleanup"""
    
    def __init__(self, log_dir="logs", retention_days=5):
        """
        Initialize the system log manager
        
        Args:
            log_dir: Directory to store log files
            retention_days: Number of days to keep log files
        """
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self.log_dir.mkdir(exist_ok=True)
        
        # Create today's log filename
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_filename = self.log_dir / f"run_system.log.{today}"
        
        # Setup logging
        self._setup_logging()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # File handler for comprehensive logging
        file_handler = logging.FileHandler(self.log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler for real-time output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Store handlers for later cleanup
        self.file_handler = file_handler
        self.console_handler = console_handler
        
        # Log system startup
        logging.info("="*80)
        logging.info(f"SYSTEM LOG MANAGER INITIALIZED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Log file: {self.log_filename}")
        logging.info(f"Retention policy: {self.retention_days} days")
        logging.info("="*80)
    
    def _start_cleanup_thread(self):
        """Start background thread for automatic log cleanup"""
        cleanup_thread = threading.Thread(target=self._cleanup_old_logs, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_logs(self):
        """Remove log files older than retention_days"""
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Find all run_system log files
            log_pattern = str(self.log_dir / "run_system.log.*")
            log_files = glob.glob(log_pattern)
            
            deleted_count = 0
            for log_file in log_files:
                try:
                    # Extract date from filename
                    filename = os.path.basename(log_file)
                    if filename.startswith("run_system.log."):
                        date_str = filename.replace("run_system.log.", "")
                        try:
                            file_date = datetime.strptime(date_str, "%Y-%m-%d")
                            
                            # Delete if older than retention period
                            if file_date < cutoff_date:
                                os.remove(log_file)
                                deleted_count += 1
                                logging.info(f"Deleted old log file: {log_file}")
                        except ValueError:
                            # Skip files with invalid date format
                            continue
                except Exception as e:
                    logging.warning(f"Error processing log file {log_file}: {e}")
            
            if deleted_count > 0:
                logging.info(f"Log cleanup completed: {deleted_count} old files deleted")
            
        except Exception as e:
            logging.error(f"Error during log cleanup: {e}")
    
    def redirect_stdout_stderr(self):
        """Redirect stdout and stderr to capture all print statements"""
        # Create a custom writer that logs everything
        class LogWriter:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level
                self.buffer = ""
            
            def write(self, message):
                # Buffer the message
                self.buffer += message
                
                # Process complete lines
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line.strip():  # Only log non-empty lines
                        self.logger.log(self.level, line.strip())
                
                # Also write to original stdout for real-time display
                sys.__stdout__.write(message)
                sys.__stdout__.flush()
            
            def flush(self):
                if self.buffer.strip():
                    self.logger.log(self.level, self.buffer.strip())
                    self.buffer = ""
                sys.__stdout__.flush()
        
        # Create logger for stdout/stderr capture
        capture_logger = logging.getLogger('system_output')
        
        # Redirect stdout and stderr
        sys.stdout = LogWriter(capture_logger, logging.INFO)
        sys.stderr = LogWriter(capture_logger, logging.ERROR)
    
    def log_system_info(self):
        """Log system information and startup details"""
        import platform
        import psutil
        
        logging.info("SYSTEM INFORMATION:")
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Python: {platform.python_version()}")
        logging.info(f"CPU Count: {psutil.cpu_count()}")
        logging.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        logging.info(f"Working Directory: {os.getcwd()}")
        logging.info(f"Process ID: {os.getpid()}")
        logging.info("-" * 80)
    
    def log_startup_banner(self):
        """Log startup banner with timestamp"""
        banner = f"""
{'='*80}
    NEWS & PRICE MONITORING SYSTEM - ZERO-LAG ARCHITECTURE
    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Log File: {self.log_filename}
{'='*80}
"""
        logging.info(banner)
    
    def log_shutdown_banner(self):
        """Log shutdown banner with timestamp"""
        banner = f"""
{'='*80}
    SYSTEM SHUTDOWN
    Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Log File: {self.log_filename}
{'='*80}
"""
        logging.info(banner)
    
    def cleanup(self):
        """Cleanup logging handlers"""
        try:
            # Restore original stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            
            # Close file handler
            if hasattr(self, 'file_handler'):
                self.file_handler.close()
            
            logging.info("Log manager cleanup completed")
        except Exception as e:
            print(f"Error during log manager cleanup: {e}")

class TeeOutput:
    """Helper class to duplicate output to both console and log file"""
    
    def __init__(self, *files):
        self.files = files
    
    def write(self, message):
        for file in self.files:
            file.write(message)
            file.flush()
    
    def flush(self):
        for file in self.files:
            file.flush()

def setup_comprehensive_logging(log_dir="logs", retention_days=5):
    """
    Setup comprehensive logging for the run_system script
    
    Args:
        log_dir: Directory to store log files
        retention_days: Number of days to keep log files
    
    Returns:
        SystemLogManager instance
    """
    log_manager = SystemLogManager(log_dir, retention_days)
    
    # Log system startup information
    log_manager.log_startup_banner()
    log_manager.log_system_info()
    
    # Redirect stdout/stderr to capture all output
    log_manager.redirect_stdout_stderr()
    
    return log_manager

if __name__ == "__main__":
    # Test the log manager
    log_manager = setup_comprehensive_logging()
    
    print("Testing comprehensive logging...")
    logging.info("This is a test log message")
    logging.warning("This is a test warning")
    logging.error("This is a test error")
    
    print("Regular print statement - should be captured")
    
    time.sleep(2)
    log_manager.cleanup() 