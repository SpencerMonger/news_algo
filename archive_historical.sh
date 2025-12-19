#!/bin/bash
#
# Archive Breaking News to Historical Table
# Cron: 32 14 * * 1-5 /home/synk/Development/newshead/archive_historical.sh
#
# This script backs up data from breaking_news → breaking_news_historical
# with alert enrichment from news_alert table.
#

SCRIPT_DIR="/home/synk/Development/newshead"
LOG_FILE="$SCRIPT_DIR/logs/archive_historical.log"
PYTHON_SCRIPT="$SCRIPT_DIR/testfiles/create_historical_table.py"

# Ensure log directory exists
mkdir -p "$SCRIPT_DIR/logs"

# Log start
echo "========================================" >> "$LOG_FILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting historical archive" >> "$LOG_FILE"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Run the archive script with --recreate to get fresh data
python3 "$PYTHON_SCRIPT" --recreate >> "$LOG_FILE" 2>&1

# Capture exit code
EXIT_CODE=$?

# Log result
if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Archive completed successfully" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Archive FAILED with exit code $EXIT_CODE" >> "$LOG_FILE"
fi

echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $EXIT_CODE

