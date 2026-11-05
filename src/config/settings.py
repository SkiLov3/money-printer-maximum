from __future__ import annotations

import copy
import logging
import os
import pathlib
from typing import Any

import yaml

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = "config/trading_stack.yaml"
_CONFIG_CACHE: dict[tuple[str, str, float], dict[str, Any]] = {}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_stack_config(path: str | None = None, profile: str | None = None) -> dict[str, Any]:
    config_path = path or os.getenv("STACK_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    if not os.path.exists(config_path):
        return {}
    selected_profile = profile or os.getenv("STACK_CONFIG_PROFILE", "")
    cache_key = (os.path.abspath(config_path), selected_profile, os.path.getmtime(config_path))
    cached = _CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if selected_profile:
        profiles = loaded.get("profiles", {})
        if selected_profile not in profiles:
            raise ValueError(f"Unknown stack config profile '{selected_profile}'")
        loaded = _merge_dict(loaded, profiles.get(selected_profile, {}))
    _CONFIG_CACHE.clear()
    _CONFIG_CACHE[cache_key] = copy.deepcopy(loaded)
    return loaded


def get_symbol_brackets(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("symbol_brackets", {}))


def get_commissions_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("commissions", {}))


def get_strategy_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("strategies", {}))


def get_optimizer_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("optimizer", {}))


def get_adaptive_optimizer_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("adaptive_optimizer", {}))


def get_feature_flags(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("feature_flags", {}))


def get_routing_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("routing", {}))


def get_zone_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("zones", {}))


def get_health_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config.get("health", {}))


def _get_dict_config(config: dict[str, Any], key: str, fallback_key: str | None = None) -> dict[str, Any]:
    """Helper to safely fetch a nested dictionary configuration with an optional fallback key, preventing None propagation."""
    val = config.get(key)
    if val is None and fallback_key:
        val = config.get(fallback_key)
    return copy.deepcopy(val) if isinstance(val, dict) else {}


def get_discovery_config(config: dict[str, Any]) -> dict[str, Any]:
    opt = config.get("optimization")
    if isinstance(opt, dict):
        val = opt.get("discovery")
        if isinstance(val, dict):
            return copy.deepcopy(val)
    return _get_dict_config(config, "discovery")


def get_meta_optimizer_config(config: dict[str, Any]) -> dict[str, Any]:
    return _get_dict_config(config, "meta_optimizer", "adaptive_optimizer")


def get_mega_vision_config(config: dict[str, Any]) -> dict[str, Any]:
    # Resolve absolute path relative to settings.py
    base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    mv_path = base_dir / "config" / "mega_vision.yaml"
    
    if mv_path.exists():
        try:
            with open(mv_path, "r", encoding="utf-8") as h:
                data = yaml.safe_load(h)
                if isinstance(data, dict):
                    return copy.deepcopy(data)
                logger.warning(f"mega_vision.yaml loaded, but content is not a dictionary: {type(data)}")
        except yaml.YAMLError as e:
            logger.warning(f"YAML syntax error in mega_vision.yaml: {e}")
        except Exception as e:
            logger.error(f"Error loading mega_vision.yaml: {e}")
            
    return _get_dict_config(config, "mega_vision")


def get_database_config(config: dict[str, Any]) -> dict[str, Any]:
    return _get_dict_config(config, "database")


def get_data_config(config: dict[str, Any]) -> dict[str, Any]:
    return _get_dict_config(config, "data")


def get_learning_review_config(config: dict[str, Any]) -> dict[str, Any]:
    opt = config.get("optimization")
    if isinstance(opt, dict):
        val = opt.get("learning_review")
        if isinstance(val, dict):
            return copy.deepcopy(val)
    return _get_dict_config(config, "learning_review")


def get_combine_rules(config: dict[str, Any], key: str = "topstep_50k") -> dict[str, Any]:
    combine = config.get("combine", {})
    rules = combine.get(key, {})
    return copy.deepcopy(rules)


def get_combine_rule_keys(config: dict[str, Any]) -> list[str]:
    combine = config.get("combine", {})
    if not isinstance(combine, dict):
        return []
    return sorted(str(key) for key in combine)
