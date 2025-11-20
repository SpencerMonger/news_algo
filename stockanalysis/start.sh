#!/bin/bash
# Stock Analysis Pipeline Runner
# Executes the complete stock analysis workflow in a screen session

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Screen session name
SCREEN_SESSION="stock_analysis_$(date +%Y%m%d_%H%M%S)"

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo -e "${RED}Error: 'screen' is not installed${NC}"
    echo -e "${YELLOW}Install it with: sudo apt-get install screen${NC}"
    exit 1
fi

# Check if we're already inside a screen session
if [ -n "$STY" ]; then
    echo -e "${YELLOW}Already running inside screen session: $STY${NC}"
    echo -e "${YELLOW}Running directly without creating nested screen...${NC}"
    INSIDE_SCREEN=true
else
    INSIDE_SCREEN=false
fi

# If not in screen, start a new screen session
if [ "$INSIDE_SCREEN" = false ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Stock Analysis Pipeline${NC}"
    echo -e "${BLUE}Starting in screen session: $SCREEN_SESSION${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}To reattach to this session, run:${NC}"
    echo -e "${GREEN}  screen -r $SCREEN_SESSION${NC}"
    echo -e "${GREEN}To detach from the session, press: Ctrl+A, then D${NC}"
    echo ""
    sleep 2
    
    # Start screen session and run this script inside it
    screen -dmS "$SCREEN_SESSION" bash -c "STY=inside $0 $@"
    
    # Attach to the screen session
    screen -r "$SCREEN_SESSION"
    exit 0
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stock Analysis Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if Python script exists
if [ ! -f "$SCRIPT_DIR/run_analysis.py" ]; then
    echo -e "${RED}Error: run_analysis.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Check for virtual environment
if [ -d "$PROJECT_ROOT/news-env" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/news-env/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo -e "${YELLOW}Warning: No virtual environment found${NC}"
    echo -e "${YELLOW}Using system Python${NC}"
fi

# Check if .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${RED}Error: .env file not found in $PROJECT_ROOT${NC}"
    echo -e "${YELLOW}Make sure you have ANTHROPIC_API_KEY and CLICKHOUSE credentials configured${NC}"
    exit 1
fi

# Run the Python script with all arguments passed through
echo -e "${GREEN}Starting pipeline...${NC}"
echo ""

python3 "$SCRIPT_DIR/run_analysis.py" "$@"

# Capture exit code
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Pipeline completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Pipeline failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE

