"""A module for loading and accessing configuration data from YAML files."""

import logging.config
import os
from logging import Logger
from typing import Any, TypeVar, overload

import yaml
from pydotenv import Environment

T = TypeVar("T")


class Config:
    """A class for loading and accessing configuration data from YAML files."""

    def __init__(self, *filepaths: str | None):
        """Initialize the Config object."""
        env = Environment().get("ENV", "development")
        self._filepaths = [
            filepath
            for filepath in [
                "config/default.yml",
                f"config/{env}.yml",
                "config/local.yml",
                f"config/{env}-local.yml",
            ]
            if os.path.isfile(filepath)
        ]
        if filepaths:
            self._filepaths.extend([filepath for filepath in filepaths if filepath and os.path.isfile(filepath)])

        self._config_data: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}

    def _load_yaml_file(self, filepath: str) -> dict:
        if not os.path.isfile(filepath):
            return {}

        with open(filepath, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _check_and_update(self, filepath: str) -> None:
        timestamp = os.path.getmtime(filepath)

        if filepath not in self._timestamps or self._timestamps[filepath] != timestamp:
            self._config_data[filepath] = self._load_yaml_file(filepath)
            self._timestamps[filepath] = timestamp

    @overload
    def get(self, path: str) -> Any | None:
        pass

    @overload
    def get(self, path: str, default=None, required=True) -> Any:
        pass

    @overload
    def get(self, path: str, default: T) -> T:
        pass

    def get(self, path: str, default: Any = None, required=False) -> Any:
        """Get a value from the configuration data."""
        keys = path.split(".")
        result = None

        for filepath in reversed(self._filepaths):
            self._check_and_update(filepath)
            config_data = self._config_data.get(filepath, {})
            current = config_data

            for key in keys:
                if not current or key not in current:
                    break
                current = current[key]
            else:
                if result is None:
                    result = current
                elif isinstance(result, dict) and isinstance(current, dict):
                    result = self._deep_merge(result, current)

        if result is None:
            if required:
                raise KeyError(f"Missing required configuration value: {path}")
            else:
                return default

        return result

    def _deep_merge(self, source: dict, destination: dict) -> dict:
        """Recursively merge two dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict):
                destination[key] = self._deep_merge(destination.get(key, {}), value)
            else:
                destination[key] = value
        return destination

    def get_logger(self, name: str | None = None) -> Logger:
        """Get a logger from the configuration data."""
        logging.config.dictConfig(self.get("log", dict[str, Any]()))
        return logging.getLogger(name)

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the Config class."""
        if not hasattr(cls, "_instance"):
            cls._instance = Config()
        return cls._instance
