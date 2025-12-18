# Core Files - Do Not Move

These files must remain in the project root directory (`/home/synk/Development/newshead/`) for the main monitoring system to function correctly.

## Entry Points

| File | Purpose |
|------|---------|
| `run_system.py` | Main Python orchestrator - starts all monitoring processes |
| `start_newshead.sh` | Startup script - runs `run_system.py` in screen session |
| `kill_newshead.sh` | Shutdown script - terminates all related processes |

## Core Monitoring Modules

| File | Purpose | Used By |
|------|---------|---------|
| `benzinga_websocket.py` | Real-time news via Benzinga WebSocket API | `run_system.py` |
| `price_checker.py` | Hybrid WebSocket + REST price monitoring via Polygon | `run_system.py` (subprocess) |
| `sentiment_service.py` | Claude AI sentiment analysis with load balancing | `run_system.py`, `benzinga_websocket.py` |
| `clickhouse_setup.py` | ClickHouse database management and triggers | All modules |
| `finviz_scraper.py` | Scrapes low-float stock list from Finviz Elite | `run_system.py` |
| `log_manager.py` | Comprehensive logging setup | `run_system.py` |

## Deprecated (but still referenced)

| File | Purpose | Notes |
|------|---------|-------|
| `web_scraper.py` | Legacy web scraping | Only used when `--socket` flag is NOT passed |

## Required Directories

| Directory | Purpose |
|-----------|---------|
| `triggers/` | Inter-process communication via JSON files |
| `logs/` | Log file storage |

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables (API keys, database credentials) |
| `requirements.txt` | Python dependencies |

---

## Safe to Reorganize

The following can be moved to subfolders without affecting the main monitoring system:

- `testfiles/` - Already organized
- `tests/` - Already organized
- `old_code/` - Already organized
- `examples/` - Already organized
- `Backtesting/` - Already organized (separate pipeline)
- `stockanalysis/` - Already organized (separate pipeline)
- `dowjones/` - Already organized
- `models/` - Already organized
- Standalone utility scripts (e.g., `check_db.py`, `drop_table.py`, `log_viewer.py`)
- Documentation files (`.md` files except this one)

