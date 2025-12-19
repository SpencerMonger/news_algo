# NewsHead - Real-Time News & Price Monitoring System

A real-time stock market news monitoring and price tracking system. Detects breaking news via Benzinga WebSocket, analyzes sentiment using Claude AI, tracks prices via IBKR TWS API, and generates alerts.

## Prerequisites

- **Python 3.8+**
- **ClickHouse** database (running and accessible)
- **IBKR TWS or Gateway** (running for price data)
- **GNU Screen** (`sudo apt install screen`)

### Required Accounts/API Keys

| Service | Purpose | Required |
|---------|---------|----------|
| ClickHouse | Database storage | Yes |
| IBKR TWS | Real-time price data | Yes |
| Benzinga | WebSocket news feed | Yes |
| Anthropic (Claude) | Sentiment analysis | Yes |
| Finviz Elite | Ticker universe updates | Optional |

## Installation

### 1. Clone & Create Virtual Environment

```bash
git clone <repository-url>
cd newshead
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install IBKR API (Manual Step)

The `ibapi` package is **not on PyPI**. You must install it manually:

**Option A**: Download from IBKR website
1. Download TWS API from [Interactive Brokers](https://interactivebrokers.github.io/)
2. Extract and navigate to `pythonclient/`
3. Run `python setup.py install`

**Option B**: Copy from existing installation
```bash
cp -r /path/to/existing/site-packages/ibapi ./venv/lib/python3.*/site-packages/
```

### 4. Install Playwright Browser

```bash
playwright install chromium
```

### 5. Configure Environment

```bash
cp env_template.txt .env
```

Edit `.env` with your credentials:

```bash
# IBKR Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497                # 7497=paper, 7496=live
IBKR_CLIENT_ID=10

# ClickHouse Database
CLICKHOUSE_HOST=your_host
CLICKHOUSE_HTTP_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_password
CLICKHOUSE_DATABASE=News
CLICKHOUSE_SECURE=true

# Benzinga (for WebSocket news)
BENZINGA_API_KEY=your_api_key

# Sentiment Analysis (Claude)
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_API_KEY2=optional_second_key    # for load balancing
ANTHROPIC_API_KEY3=optional_third_key

# Finviz Elite (optional - for ticker list updates)
FINVIZ_EMAIL=your_email
FINVIZ_PASSWORD=your_password
```

## Running the System

### Step 1: Run Stock Analysis Pipeline (First Time)

This populates the ticker universe and generates strength scores for alert prioritization:

```bash
./stockanalysis/start.sh
```

Or run directly:
```bash
python3 stockanalysis/run_analysis.py
```

### Step 2: Start Main Monitoring System

```bash
./start_newshead.sh
```

This starts the system in a `screen` session. To view logs:
```bash
screen -r newshead
```

To detach from screen: `Ctrl+A`, then `D`

### Stop the System

```bash
./kill_newshead.sh
```

## Command Line Options

Run directly with Python for more control:

```bash
# Standard run with WebSocket (recommended)
python3 run_system.py --socket

# Process any ticker (bypass ticker list filtering)
python3 run_system.py --socket --any

# Disable sentiment analysis (testing)
python3 run_system.py --socket --no-sentiment

# Process older articles (testing)
python3 run_system.py --socket --enable-old
```

## Directory Structure

```
newshead/
├── run_system.py           # Main orchestrator
├── benzinga_websocket.py   # Real-time news via WebSocket
├── price_checker.py        # IBKR price monitoring
├── sentiment_service.py    # Claude AI sentiment analysis
├── clickhouse_setup.py     # Database management
├── stockanalysis/          # Stock analysis pipeline
│   ├── start.sh            # Pipeline runner
│   ├── run_analysis.py     # Main pipeline script
│   └── ...
├── triggers/               # File-based inter-process communication
├── logs/                   # System logs
└── Backtesting/            # Historical backtesting tools
```

## Logs

Log files are stored in `logs/`:
- `logs/run_system.log.YYYY-MM-DD` - Main system logs
- `logs/articles/article_tracking.log` - Article processing logs

## Troubleshooting

### IBKR Connection Issues
- Ensure TWS/Gateway is running and API connections are enabled
- Check port: 7497 for paper trading, 7496 for live
- Verify client ID isn't already in use

### ClickHouse Connection Issues
- Verify ClickHouse server is running
- Check credentials in `.env`
- Test connection: `python3 -c "from clickhouse_setup import ClickHouseManager; ch = ClickHouseManager(); ch.connect(); print('Connected!')"`

### Screen Session Issues
- Install screen: `sudo apt install screen`
- List sessions: `screen -ls`
- Kill stuck session: `screen -X -S newshead quit`

## Architecture

See [Architecture.md](Architecture.md) for detailed system documentation.
