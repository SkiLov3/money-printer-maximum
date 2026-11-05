import asyncio
import contextlib
import logging
import os
import signal
import threading
from http.server import ThreadingHTTPServer
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

from src.agents.supervisor_agent import SupervisorAgent

load_dotenv()

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            "logs/trades.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        ),
    ],
)
logger = logging.getLogger("Main")


def _start_dashboard(supervisor):
    """Launch the dashboard HTTP server in a background daemon thread."""
    try:
        from dashboard import HOST, PASSWORD, PORT, VERSION, Handler

        # Inject supervisor into the Handler class
        Handler.supervisor = supervisor

        server = ThreadingHTTPServer((HOST, PORT), Handler)
        auth_note = "password protected" if PASSWORD else "no auth - set DASHBOARD_PASSWORD for remote use"
        mode = os.getenv("MODE", "simulation").strip().upper()
        symbols = os.getenv("SYMBOL", "NQ")

        # Banner is best-effort: Windows' default cp1252 console can't encode emoji/ANSI in
        # some shells. Don't let a print encoding error kill the dashboard thread.
        try:
            mode_color = "\033[92m" if mode == "SIMULATION" else "\033[91m"  # green sim, red live
            reset = "\033[0m"
            bold = "\033[1m"
            print(f"\n{'=' * 55}")
            print(f"  TopstepX Swarm is running! (v{VERSION})")
            print(f"  Mode:          {mode_color}{bold}{mode}{reset}")
            print(f"  Symbols:       {bold}{symbols}{reset}")
            print(f"{'=' * 55}")
            print(f"  Dashboard  ->  http://localhost:{PORT}")
            print(f"  Auth:          {auth_note}")
            print("  Logs:          logs/trades.log")
            print("  Press Ctrl+C to stop the bot.\n")
        except Exception as banner_err:
            logger.info(f"Dashboard banner skipped ({banner_err}); server still starting on port {PORT}.")

        logger.info(f"Dashboard listening on http://{HOST}:{PORT} (mode={mode}, symbols={symbols})")
        server.serve_forever()
    except Exception as e:
        logger.warning(f"Dashboard failed to start: {e}")


def _require_stack_config() -> None:
    """Fail fast if the live stack config is missing (it is git-ignored on purpose).

    load_stack_config() silently returns {} for a missing file, which would make the
    bot run on code defaults instead of the intended strategy parameters.
    """
    from src.config.settings import DEFAULT_CONFIG_PATH

    config_path = os.getenv("STACK_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    if not os.path.exists(config_path):
        logger.critical(
            "Stack config not found at '%s'. Create it from the template:\n"
            "    cp config/trading_stack.example.yaml %s\n"
            "then tune it for your own research. Refusing to start on built-in defaults.",
            config_path,
            config_path,
        )
        raise SystemExit(1)


async def main():
    _require_stack_config()
    symbols_str = os.getenv("SYMBOL") or os.getenv("SYMBOLS", "NQ")
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    account_id = os.getenv("PROJECTX_ACCOUNT_ID", "DEMO-123")

    # Start Bot Swarm
    supervisor = SupervisorAgent(symbols, account_id)

    # Start dashboard in background — pass supervisor instance
    t = threading.Thread(target=_start_dashboard, args=(supervisor,), daemon=True)
    t.start()

    # Async-Safe Signal Handling for Clean Shutdown
    loop = asyncio.get_running_loop()

    def handle_exit():
        logger.info("SIGINT/SIGTERM received. Initiating graceful shutdown...")
        supervisor.stop()

    # add_signal_handler isn't implemented on Windows ProactorEventLoop — fall back to
    # signal.signal() so the bot still installs handlers there. SIGTERM doesn't exist on
    # Windows; gracefully ignore that case.
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_exit)
    except NotImplementedError:
        for sig_name in ("SIGINT", "SIGTERM"):
            sig_obj = getattr(signal, sig_name, None)
            if sig_obj is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig_obj, lambda *_: handle_exit())

    try:
        logger.info("--- TopstepX Trading Swarm Starting ---")
        await supervisor.start()
    except Exception as e:
        logger.critical(f"FATAL Swarm Error: {e}", exc_info=True)
    finally:
        logger.info("Bot swarm deactivated. Shutting down event loop.")
        # Final safety stop
        if "supervisor" in locals():
            supervisor.stop()
        logger.info("Safe to exit.")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt, asyncio.exceptions.CancelledError):
        asyncio.run(main())

