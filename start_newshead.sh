#!/bin/bash

# Define home and project directories with absolute paths
HOME="/home/synk"
PROJECT_DIR="$HOME/Development/newshead"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Log file for today
LOG_FILE="$LOG_DIR/start_system_$(date +\%Y\%m\%d).log"
EXEC_LOG="$LOG_DIR/script_execution.log"

# Log the execution
echo "=========================" >> "$EXEC_LOG"
echo "Script executed at $(date) [UTC]" >> "$EXEC_LOG"
echo "User: $(whoami)" >> "$EXEC_LOG"
echo "PWD: $(pwd)" >> "$EXEC_LOG"
echo "HOME: $HOME" >> "$EXEC_LOG"
echo "=========================" >> "$EXEC_LOG"

# Make sure the kill script is executable
chmod +x "$PROJECT_DIR/kill_newshead.sh"

# Change to the project directory
cd "$PROJECT_DIR" || {
    echo "$(date) [UTC]: ERROR: Could not change to project directory $PROJECT_DIR" >> "$LOG_FILE"
    echo "ERROR: Could not change to project directory"
    exit 1
}
echo "Changed directory to $(pwd)" >> "$EXEC_LOG"

# Activate the virtual environment
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
    echo "$(date) [UTC]: Activated .venv virtual environment" >> "$LOG_FILE"
    echo "Activated .venv" >> "$EXEC_LOG"
elif [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "$(date) [UTC]: Activated venv virtual environment" >> "$LOG_FILE"
    echo "Activated venv" >> "$EXEC_LOG"
else
    echo "$(date) [UTC]: WARNING: No virtual environment found" >> "$LOG_FILE"
    echo "WARNING: No virtual environment found" >> "$EXEC_LOG"
fi

# Log Python path and version
PYTHON_BIN=$(which python3)
echo "Using Python: $PYTHON_BIN" >> "$EXEC_LOG"
$PYTHON_BIN --version >> "$EXEC_LOG" 2>&1

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
echo "Set PYTHONPATH to include $PROJECT_DIR" >> "$EXEC_LOG"

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "$(date) [UTC]: ERROR: screen is not installed" >> "$LOG_FILE"
    echo "ERROR: screen is not installed. Please install it with: sudo apt-get install screen"
    exit 1
fi

# Kill existing screen session if it exists
if screen -ls | grep -q "newshead"; then
    echo "$(date) [UTC]: newshead screen session already exists, killing old one first" >> "$LOG_FILE"
    screen -X -S newshead quit
    sleep 2
    
    # Double check if it was killed
    if screen -ls | grep -q "newshead"; then
        echo "$(date) [UTC]: WARNING: Could not kill existing screen session" >> "$LOG_FILE"
        # Try more aggressively to kill the session
        screen_pid=$(screen -ls | grep newshead | awk '{print $1}' | cut -d. -f1)
        if [ ! -z "$screen_pid" ]; then
            echo "$(date) [UTC]: Attempting to kill screen session with PID $screen_pid" >> "$LOG_FILE"
            kill -9 "$screen_pid"
            sleep 1
        fi
    fi
fi

# Additional check for any running Python processes related to run_system.py
system_pids=$(pgrep -f "python.*run_system.py")
if [ ! -z "$system_pids" ]; then
    echo "$(date) [UTC]: Found related Python processes. Attempting to terminate: $system_pids" >> "$LOG_FILE"
    kill $system_pids
    sleep 1
fi

# Run script in screen session with WebSocket mode enabled
echo "$(date) [UTC]: Starting run_system.py with WebSocket mode in a screen session" >> "$LOG_FILE"
cd "$PROJECT_DIR" && screen -dmS newshead $PYTHON_BIN run_system.py --socket
echo "Started screen session with name 'newshead' using WebSocket mode" >> "$EXEC_LOG"

# Give it a moment to start
sleep 2

# Verify screen session is running
if screen -ls | grep -q "newshead"; then
    echo "$(date) [UTC]: Confirmed newshead screen session is running with WebSocket mode" >> "$LOG_FILE"
    echo "SUCCESS: newshead screen session is running with WebSocket mode" >> "$LOG_FILE"
else
    echo "$(date) [UTC]: ERROR: Failed to start newshead screen session" >> "$LOG_FILE"
    echo "ERROR: Failed to start newshead screen session" >> "$LOG_FILE"
    exit 1
fi

# List available screen sessions for reference
echo "Available screen sessions:" >> "$LOG_FILE"
screen -ls >> "$LOG_FILE"
