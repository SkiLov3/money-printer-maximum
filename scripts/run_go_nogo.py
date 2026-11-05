import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_database_config, load_stack_config
from src.persistence.database import DatabaseService
from src.validation.go_nogo import GoNoGoValidator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--config-file", default="config/trading_stack.yaml")
    args = parser.parse_args()

    stack_config = load_stack_config(args.config_file)
    database = DatabaseService(get_database_config(stack_config))
    df = pd.read_csv(args.data_file)
    validator = GoNoGoValidator(args.symbol, database=database)
    result = validator.validate(df, stack_config.get("strategies", {}))
    print({"verdict": result.verdict, "metrics": result.metrics, "recommendations": result.recommendations})


if __name__ == "__main__":
    main()
