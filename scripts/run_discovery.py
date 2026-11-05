import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_database_config, get_discovery_config, load_stack_config
from src.discovery.service import DiscoveryService
from src.persistence.database import DatabaseService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--config-file", default="config/trading_stack.yaml")
    args = parser.parse_args()

    stack_config = load_stack_config(args.config_file)
    database = DatabaseService(get_database_config(stack_config))
    discovery = DiscoveryService(get_discovery_config(stack_config), database=database)
    df = pd.read_csv(args.data_file)
    decision = discovery.evaluate_trade(
        args.symbol,
        df,
        {"symbol": args.symbol, "strategy_name": "manual_discovery_probe", "quantity": 1, "side": "Buy"},
    )
    print(decision)


if __name__ == "__main__":
    main()
