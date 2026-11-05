import os
import sys
import logging
from dotenv import load_dotenv
from src.api.projectx_client import ProjectXClient

# Use the same logging config as main
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("HealthCheck")

def run_health_check():
    """Verifies environment, credentials, and API connectivity before bot startup."""
    load_dotenv()
    
    logger.info("--- TopstepX Pre-Flight Health Check ---")
    
    # 1. Check Environment Variables
    required_vars = ["PROJECTX_API_KEY", "TOPSTEPX_USERNAME", "PROJECTX_ACCOUNT_ID", "SYMBOL"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False
    logger.info("Environment variables: OK")

    # 2. Test API Connectivity & Login
    client = ProjectXClient()
    if not client.login():
        logger.error("API Connectivity/Authentication: FAILED")
        return False
    logger.info("API Connectivity & Authentication: OK")

    # 3. Verify Account Status
    status = client.get_account_status()
    if not status:
        logger.error(f"Account ID '{client.account_id}': NOT FOUND or UNAUTHORIZED")
        return False
    
    balance = status.get('balance', 0)
    pnl = status.get('dailyPnL', 0)
    logger.info(f"Account '{client.account_id}' Verified | Balance: ${balance:,.2f} | Daily PnL: ${pnl:,.2f}")

    # 4. Symbol Resolution Check
    symbols = os.getenv("SYMBOL", "MNQ").split(",")
    for sym in symbols:
        sym = sym.strip()
        resolved = client.resolve_symbol(sym)
        if not resolved:
            logger.error(f"Symbol Resolution for '{sym}': FAILED")
            return False
        logger.info(f"Symbol '{sym}' -> '{resolved}': OK")

    logger.info("--- Health Check: PASSED ---")
    return True

if __name__ == "__main__":
    if run_health_check():
        sys.exit(0)
    else:
        sys.exit(1)
