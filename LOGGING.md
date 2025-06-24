# Comprehensive Logging System

This document explains the comprehensive logging system implemented for the `run_system` script with automatic rotation and cleanup.

## Overview

The logging system captures **ALL** output from the `run_system` script including:
- All log messages from the main system
- All print statements and console output
- Output from subprocess (price_checker.py)
- System information and startup/shutdown details
- Error traces and debugging information

## Features

- **Daily Log Files**: One log file per day (`run_system.log.YYYY-MM-DD`)
- **Automatic Cleanup**: Keeps only the most recent 5 days of logs
- **Comprehensive Capture**: All stdout/stderr is captured and logged
- **Real-time Console Output**: Still displays output to console while logging
- **System Information**: Logs platform, memory, CPU info on startup
- **Process Isolation Logging**: Captures output from separate price checker process

## Files Created

### Core Logging Files
- `log_manager.py` - Main logging system with automatic rotation and cleanup
- `run_system.py` - Modified to use comprehensive logging
- `log_viewer.py` - Utility to view and manage log files
- `run_system_cron.sh` - Cron-compatible wrapper script

### Log Files (in `logs/` directory)
- `run_system.log.2025-01-15` - Today's complete system log
- `run_system.log.2025-01-14` - Yesterday's log
- etc. (keeps 5 days automatically)

## Usage

### Running with Comprehensive Logging

#### Direct Python Execution
```bash
python run_system.py
```
All output will be captured to `logs/run_system.log.YYYY-MM-DD`

#### Using Cron Wrapper (Recommended for Crontab)
```bash
./run_system_cron.sh
```
This ensures proper environment setup and logging for cron jobs.

### Viewing Logs

#### Using the Log Viewer Utility
```bash
# List all available log files
python log_viewer.py list

# View today's log
python log_viewer.py today

# View specific log file (by number from list)
python log_viewer.py view 1

# View specific log file (by date)
python log_viewer.py view 2025-01-15

# View last 100 lines of today's log
python log_viewer.py today --lines 100

# Follow today's log in real-time (like tail -f)
python log_viewer.py today --follow

# Filter log entries containing "ERROR"
python log_viewer.py today --filter ERROR

# View log statistics
python log_viewer.py stats
```

#### Direct File Access
```bash
# View today's log
tail -f logs/run_system.log.$(date +%Y-%m-%d)

# View specific date
cat logs/run_system.log.2025-01-15

# Search for errors
grep -i error logs/run_system.log.*
```

### Log Cleanup

#### Automatic Cleanup
- Runs automatically when the system starts
- Keeps the most recent 5 days of logs
- Deletes older files to save disk space

#### Manual Cleanup
```bash
# Dry run - see what would be deleted
python log_viewer.py cleanup

# Actually delete old files
python log_viewer.py cleanup --execute

# Change retention period (e.g., 7 days)
python log_viewer.py cleanup --days 7 --execute
```

## Crontab Integration

### Setting Up Cron Job

1. **Edit your crontab:**
```bash
crontab -e
```

2. **Add entry to run daily at 6 AM:**
```bash
0 6 * * * /home/synk/Development/newshead/run_system_cron.sh >> /home/synk/Development/newshead/logs/cron.log 2>&1
```

3. **Or run with specific arguments:**
```bash
0 6 * * * /home/synk/Development/newshead/run_system_cron.sh --skip-list >> /home/synk/Development/newshead/logs/cron.log 2>&1
```

### Cron Wrapper Features

The `run_system_cron.sh` wrapper provides:
- Proper working directory setup
- Virtual environment activation
- Environment variable loading
- Process conflict detection
- Comprehensive logging
- Automatic cleanup
- Exit code handling

## Log File Format

Each log entry includes:
- **Timestamp**: `2025-01-15 14:30:45`
- **Logger Name**: Module or component name
- **Level**: INFO, WARNING, ERROR, etc.
- **Message**: The actual log content

Example:
```
2025-01-15 14:30:45 - __main__ - INFO - === Starting News & Price Monitoring System ===
2025-01-15 14:30:45 - __main__ - INFO - Setting up ClickHouse database...
2025-01-15 14:30:46 - system_output - INFO - [PRICE_CHECKER] Starting continuous price monitoring...
```

## Monitoring Log Files

### Check Current System Status
```bash
# See if system is running
python log_viewer.py today --filter "Starting\|Terminating" --lines 20

# Check for errors
python log_viewer.py today --filter "ERROR\|FATAL"

# Monitor real-time
python log_viewer.py today --follow
```

### Log File Sizes
- Typical daily log: 10-50 MB (depending on activity)
- 5-day retention: ~50-250 MB total
- Automatic cleanup prevents unlimited growth

## Troubleshooting

### No Log Files Created
- Check if `logs/` directory exists and is writable
- Verify `psutil` is installed: `pip install psutil>=5.9.0`
- Check for Python import errors

### Large Log Files
- Log files can be large due to comprehensive capture
- This is normal for complete system monitoring
- Automatic cleanup keeps only 5 days
- Adjust retention with `--days` parameter if needed

### Missing Subprocess Output
- Price checker output is captured via `[PRICE_CHECKER]` prefix
- If missing, check subprocess stdout capture in `run_system.py`

### Cron Job Not Working
- Check cron logs: `grep CRON /var/log/syslog`
- Verify script permissions: `ls -la run_system_cron.sh`
- Test wrapper manually: `./run_system_cron.sh --skip-list`
- Check environment variables are loaded

## Performance Impact

### Logging Overhead
- Minimal performance impact (~1-2% CPU)
- File I/O is buffered and asynchronous
- Real-time console output maintained

### Disk Usage
- 5-day retention typically uses 50-250 MB
- Automatic cleanup prevents runaway growth
- Monitor with `python log_viewer.py stats`

## Advanced Configuration

### Changing Retention Period
Modify in `log_manager.py`:
```python
log_manager = SystemLogManager(log_dir="logs", retention_days=7)  # 7 days instead of 5
```

### Custom Log Directory
```python
log_manager = setup_comprehensive_logging(log_dir="custom_logs", retention_days=5)
```

### Filtering Specific Output
Add filters in `log_manager.py` to exclude verbose output if needed.

## Integration with Existing Logs

The system preserves existing ClickHouse operation logs:
- `clickhouse_operations.log.YYYY-MM-DD` - Database operations
- `run_system.log.YYYY-MM-DD` - Complete system logs (new)

Both follow the same date-based naming convention for consistency. 