#!/bin/bash
# Archive breaking news to historical table
# Add to crontab with: crontab -e
# Example: 0 2 * * * /home/synk/Development/newshead/archive_historical.sh

cd /home/synk/Development/newshead

# Activate virtual environment if it exists
if [ -d "news-env" ]; then
    source news-env/bin/activate
fi

# Run the historical archive script
python3 create_historical_table.py

# Exit with the same status as the Python script
exit $?

