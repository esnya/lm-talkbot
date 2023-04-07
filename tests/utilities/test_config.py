"""Tests for the Config class."""

import unittest

from talkbot.utilities.config import Config


class TestConfig(unittest.TestCase):
    """Tests for the Config class."""

    def test_get(self) -> None:
        """Test the get method."""
        config = Config("tests/config/test1.yml", "tests/config/test2.yml")

        self.assertEqual(config.get("database.host"), "example2.com")
        self.assertEqual(config.get("database.port"), 1234)
        self.assertEqual(config.get("database.user"), "user2")
        self.assertEqual(config.get("database.password"), "pass1")

        self.assertEqual(config.get("extra.key"), "value")
        self.assertIsNone(config.get("extra.unknown"))

        self.assertEqual(config.get("log.version"), 1)
        self.assertIsNone(config.get("log.unknown"))

    def test_get_with_default(self) -> None:
        """Test the get method with a default value."""

        config = Config("config/test1.yml", "config/test2.yml")

        self.assertEqual(config.get("extra.unknown", "default_value"), "default_value")
