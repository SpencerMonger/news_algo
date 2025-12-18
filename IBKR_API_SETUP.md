# IBKR API and PDM Setup Documentation

## Current Working Configuration

This document describes the working IBKR Python API setup used in the newstrader project, which can be replicated for other projects.

## IBKR Python API Installation

### Version
- **Active Version:** `10.30.1`
- **Location:** `/home/synk/Development/tradehead/.venv/lib/python3.12/site-packages/ibapi`

### Local Source Copy
- **Version:** `10.19.1`
- **Location:** `/home/synk/Development/newstrader/lib/IBJts/source/pythonclient`

Note: The active version (10.30.1) is newer and installed via pip/PyPI, not from the local source.

## PDM Configuration

### Environment Details
- **PDM Version:** 2.23.1
- **Python Version:** 3.12
- **Virtual Environment:** `/home/synk/Development/tradehead/.venv/`
- **PDM Binary Location:** `/home/synk/.local/bin/pdm`

### Key Configuration
The project uses a **shared virtual environment** across multiple projects:
- `.pdm-python` file contains: `/home/synk/Development/tradehead/.venv/bin/python`
- This allows multiple projects to share the same Python environment and dependencies

## How to Replicate This Setup for a New Project

### Option 1: Use the Existing Shared Virtual Environment (Recommended for Compatibility)

1. **Create your new project directory:**
   ```bash
   cd /home/synk/Development/
   mkdir your-new-project
   cd your-new-project
   ```

2. **Initialize PDM and point to the shared venv:**
   ```bash
   pdm init --python /home/synk/Development/tradehead/.venv/bin/python
   ```

3. **Create a `pyproject.toml` with IBKR API dependency:**
   ```toml
   [project]
   name = "your-project-name"
   version = "0.1.0"
   dependencies = [
       "ibapi>=10.30.1",
       # Add other dependencies as needed
   ]
   requires-python = "==3.12.*"
   ```

4. **Install dependencies:**
   ```bash
   pdm install
   ```

5. **Verify IBKR API is available:**
   ```bash
   pdm run python -c "import ibapi; print(f'ibapi version: {ibapi.__version__}')"
   ```

### Option 2: Create a Fresh Local Virtual Environment

1. **Create your new project directory:**
   ```bash
   cd /home/synk/Development/
   mkdir your-new-project
   cd your-new-project
   ```

2. **Initialize PDM with a local venv:**
   ```bash
   pdm init
   # When prompted, select Python 3.12
   # Choose to create a virtualenv in .venv
   ```

3. **Install IBKR API from pip:**
   ```bash
   pdm add "ibapi>=10.30.1"
   ```

4. **Verify installation:**
   ```bash
   pdm run python -c "import ibapi; print(f'ibapi version: {ibapi.__version__}')"
   ```

## Running Python Scripts with PDM

### Command Structure
```bash
pdm run python your_script.py
```

### Example from newstrader:
```bash
pdm run python src/run_local.py --config config.yaml --port 7497
```

### In a Screen Session (for background execution):
```bash
screen -dmS session-name pdm run python your_script.py
```

## Essential IBKR API Connection Details

### Ports
- **Paper Trading:** 7497
- **Live Trading:** 7496

### Client ID
- Each connection needs a unique client ID
- The newstrader system reserves client ID 0 for manual orders
- Use different client IDs for different applications

### TWS/Gateway Requirements
- Must be running before connecting via API
- Requires API to be enabled in settings:
  - File → Global Configuration → API → Settings
  - Enable "Enable ActiveX and Socket Clients"
  - Check "Bypass Order Precautions for API Orders" (under Precautions)

## Basic IBKR API Usage Pattern

```python
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBWrapper(EWrapper):
    def __init__(self):
        super().__init__()
    
    def nextValidId(self, orderId: int):
        # Called when connection is established
        self.nextOrderId = orderId
        print(f"Connected. Next order ID: {orderId}")
    
    def tickPrice(self, reqId, tickType, price, attrib):
        # Called when price data is received
        print(f"TickPrice. Ticker ID: {reqId}, Type: {tickType}, Price: {price}")

class IBClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

# Usage
wrapper = IBWrapper()
client = IBClient(wrapper)

# Connect (host, port, client_id)
client.connect("127.0.0.1", 7497, 1)

# Start message processing thread
import threading
api_thread = threading.Thread(target=client.run, daemon=True)
api_thread.start()
```

## Important Notes

1. **Shared Virtual Environment:** The newstrader project uses a shared venv from the tradehead project. This is unconventional but works fine for development.

2. **PYTHONPATH:** If not using PDM, you may need to set PYTHONPATH:
   ```bash
   export PYTHONPATH=/path/to/your/project:$PYTHONPATH
   ```

3. **API Version:** Version 10.30.1 is from mid-2024 and includes recent bug fixes and features. It's compatible with TWS API version 10.19+.

4. **Threading:** IBKR API requires running in a separate thread. The `client.run()` method is blocking and handles the message loop.

## Verification Commands

Check your setup:
```bash
# Find ibapi location and version
pdm run python -c "import ibapi; import os; print(f'Location: {os.path.dirname(ibapi.__file__)}'); print(f'Version: {ibapi.__version__}')"

# Check PDM configuration
pdm info

# List installed packages
pdm list
```

## Additional Resources

- [TWS API Documentation](https://interactivebrokers.github.io/tws-api/)
- [Python API Reference](https://interactivebrokers.github.io/tws-api/classIBApi_1_1EClient.html)
- [API Forum](https://www.interactivebrokers.com/en/general/developers/api-developers.php)

