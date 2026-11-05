import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_database_config, get_feature_flags, load_stack_config
from src.persistence.database import DatabaseService


def main():
    config = load_stack_config()
    flags = get_feature_flags(config)
    db_config = get_database_config(config) if flags.get("database_enabled", False) else {"enabled": False}
    database = DatabaseService(db_config)
    print(database.health_check())


if __name__ == "__main__":
    main()
