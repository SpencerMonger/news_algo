#!/bin/bash

# Define home and project directories with absolute paths
HOME="/home/synk"
PROJECT_DIR="$HOME/Development/newshead"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Log file for today
LOG_FILE="$LOG_DIR/kill_system_$(date +\%Y\%m\%d).log"

echo "=========================" >> "$LOG_FILE"
echo "Kill script executed at $(date) [UTC]" >> "$LOG_FILE"
echo "User: $(whoami)" >> "$LOG_FILE"
echo "PWD: $(pwd)" >> "$LOG_FILE"
echo "=========================" >> "$LOG_FILE"

echo "$(date) [UTC]: Attempting to kill newshead screen session" >> "$LOG_FILE"

# Kill the session
screen -X -S newshead quit

# Give it a moment to terminate
sleep 2

# Double check
if screen -ls | grep -q "newshead"; then
    echo "$(date) [UTC]: WARNING: Screen session could not be killed" >> "$LOG_FILE"
    # Try more aggressively to kill the session
    screen_pid=$(screen -ls | grep newshead | awk '{print $1}' | cut -d. -f1)
    if [ ! -z "$screen_pid" ]; then
        echo "$(date) [UTC]: Attempting to kill screen session with PID $screen_pid" >> "$LOG_FILE"
        kill -9 "$screen_pid"
        sleep 1
    fi
    
    # Check again after forced kill
    if screen -ls | grep -q "newshead"; then
        echo "$(date) [UTC]: ERROR: Still could not kill the screen session" >> "$LOG_FILE"
        echo "ERROR: Failed to terminate newshead screen session" >> "$LOG_FILE"
    else
        echo "$(date) [UTC]: Successfully terminated screen session after forced kill" >> "$LOG_FILE"
        echo "SUCCESS: Terminated screen session after forced kill" >> "$LOG_FILE"
    fi
else
    echo "$(date) [UTC]: Successfully terminated screen session" >> "$LOG_FILE"
    echo "SUCCESS: Terminated screen session" >> "$LOG_FILE"
fi

# Additional check for any running Python processes related to run_system.py
system_pids=$(pgrep -f "python.*run_system.py")
if [ ! -z "$system_pids" ]; then
    echo "$(date) [UTC]: Found related Python processes. Attempting to terminate: $system_pids" >> "$LOG_FILE"
    kill $system_pids
    sleep 1
    
    # Check if processes are still running
    remaining_pids=$(pgrep -f "python.*run_system.py")
    if [ ! -z "$remaining_pids" ]; then
        echo "$(date) [UTC]: WARNING: Some processes still running. Attempting force kill: $remaining_pids" >> "$LOG_FILE"
        kill -9 $remaining_pids
        echo "WARNING: Had to force kill some Python processes" >> "$LOG_FILE"
    else
        echo "$(date) [UTC]: Successfully terminated all Python processes" >> "$LOG_FILE"
        echo "SUCCESS: All Python processes terminated" >> "$LOG_FILE"
    fi
else
    echo "$(date) [UTC]: No related Python processes found" >> "$LOG_FILE"
    echo "INFO: No related Python processes were running" >> "$LOG_FILE"
fi

# NEW: Kill isolated price_checker.py processes (these become orphaned)
echo "$(date) [UTC]: Checking for isolated price_checker.py processes" >> "$LOG_FILE"
price_checker_pids=$(pgrep -f "python.*price_checker.py")
if [ ! -z "$price_checker_pids" ]; then
    echo "$(date) [UTC]: Found isolated price_checker processes: $price_checker_pids" >> "$LOG_FILE"
    echo "$(date) [UTC]: Attempting to terminate price_checker processes" >> "$LOG_FILE"
    kill $price_checker_pids
    sleep 2
    
    # Check if processes are still running
    remaining_price_pids=$(pgrep -f "python.*price_checker.py")
    if [ ! -z "$remaining_price_pids" ]; then
        echo "$(date) [UTC]: WARNING: Some price_checker processes still running. Force killing: $remaining_price_pids" >> "$LOG_FILE"
        kill -9 $remaining_price_pids
        sleep 1
        
        # Final check
        final_price_pids=$(pgrep -f "python.*price_checker.py")
        if [ ! -z "$final_price_pids" ]; then
            echo "$(date) [UTC]: ERROR: Could not kill all price_checker processes: $final_price_pids" >> "$LOG_FILE"
            echo "ERROR: Some price_checker processes could not be terminated" >> "$LOG_FILE"
        else
            echo "$(date) [UTC]: Successfully force-killed all price_checker processes" >> "$LOG_FILE"
            echo "SUCCESS: All price_checker processes terminated after force kill" >> "$LOG_FILE"
        fi
    else
        echo "$(date) [UTC]: Successfully terminated all price_checker processes" >> "$LOG_FILE"
        echo "SUCCESS: All price_checker processes terminated" >> "$LOG_FILE"
    fi
else
    echo "$(date) [UTC]: No isolated price_checker processes found" >> "$LOG_FILE"
    echo "INFO: No isolated price_checker processes were running" >> "$LOG_FILE"
fi

# NEW: Also check for any other newshead-related Python processes
echo "$(date) [UTC]: Checking for any other newshead-related processes" >> "$LOG_FILE"
other_newshead_pids=$(pgrep -f "python.*" | xargs -I {} sh -c 'ps -p {} -o pid,cmd --no-headers' | grep -E "(web_scraper|news|newshead)" | grep -v "kill_newshead" | awk '{print $1}')
if [ ! -z "$other_newshead_pids" ]; then
    echo "$(date) [UTC]: Found other newshead-related processes: $other_newshead_pids" >> "$LOG_FILE"
    echo "$(date) [UTC]: Attempting to terminate other newshead processes" >> "$LOG_FILE"
    echo $other_newshead_pids | xargs kill
    sleep 1
    echo "$(date) [UTC]: Terminated other newshead-related processes" >> "$LOG_FILE"
    echo "SUCCESS: Other newshead processes terminated" >> "$LOG_FILE"
else
    echo "$(date) [UTC]: No other newshead-related processes found" >> "$LOG_FILE"
    echo "INFO: No other newshead processes were running" >> "$LOG_FILE"
fi

# List available screen sessions for reference
echo "Current screen sessions:" >> "$LOG_FILE"
screen -ls >> "$LOG_FILE"

# Final process check
echo "$(date) [UTC]: Final process check for any remaining newshead processes" >> "$LOG_FILE"
remaining_processes=$(ps aux | grep -E "(python.*newshead|python.*price_checker|python.*run_system|python.*web_scraper)" | grep -v grep | grep -v kill_newshead)
if [ ! -z "$remaining_processes" ]; then
    echo "$(date) [UTC]: WARNING: Some processes may still be running:" >> "$LOG_FILE"
    echo "$remaining_processes" >> "$LOG_FILE"
else
    echo "$(date) [UTC]: SUCCESS: No remaining newshead processes found" >> "$LOG_FILE"
fi

echo "=========================" >> "$LOG_FILE"
echo "Kill script completed at $(date) [UTC]" >> "$LOG_FILE"
echo "=========================" >> "$LOG_FILE"

