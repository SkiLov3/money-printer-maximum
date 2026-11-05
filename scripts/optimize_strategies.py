#!/usr/bin/env python3
import argparse
import json

import pandas as pd

from src.config.settings import get_optimizer_config, load_stack_config
from src.data.catalog import DatasetCatalog
from src.optimization.optimizer import StrategyOptimizer


def main():
    parser = argparse.ArgumentParser(description="Optimize multi-strategy configs on a CSV dataset.")
    parser.add_argument("--symbol", required=True, help="Resolved or base symbol, e.g. MNQM6")
    parser.add_argument("--file", required=True, help="CSV dataset path")
    parser.add_argument("--config", default=None, help="Path to YAML stack config (default: config/trading_stack.yaml)")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    catalog = DatasetCatalog()
    catalog.register_dataset(args.symbol, args.file, source="optimizer_input")

    stack_config = load_stack_config(args.config)
    optimizer = StrategyOptimizer(args.symbol, optimizer_config=get_optimizer_config(stack_config))
    result = optimizer.optimize_dataframe(df)
    print(json.dumps(result["best"], indent=2, default=str))


if __name__ == "__main__":
    main()
