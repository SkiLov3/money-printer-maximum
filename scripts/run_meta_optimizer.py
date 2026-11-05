import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_database_config, load_stack_config
from src.discovery.meta_optimizer import MetaOptimizerService
from src.persistence.database import DatabaseService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--config-file", default="config/trading_stack.yaml")
    args = parser.parse_args()

    stack_config = load_stack_config(args.config_file)
    database = DatabaseService(get_database_config(stack_config))
    service = MetaOptimizerService(args.symbol, args.data_file, config_path=args.config_file, database=database)
    result = service.run_epoch(epoch=args.epoch or None, trials_per_epoch=args.trials or None)
    print(result)


if __name__ == "__main__":
    main()
