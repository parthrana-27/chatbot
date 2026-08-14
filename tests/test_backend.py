import os
import sys
import unittest

# Ensure root folder is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set dummy HF token for testing environment if not present
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_dummy_token_for_testing"

from backend.config import get_hf_token, DEFAULT_MODEL, PORT, HOST


class TestBackendConfig(unittest.TestCase):

    def test_get_hf_token_override(self):
        """Test that passing an explicit token takes precedence."""
        override = "hf_custom_token"
        result = get_hf_token(override)
        self.assertEqual(result, override)

    def test_get_hf_token_env_fallback(self):
        """Test fallback to environment variable when override is empty."""
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_env_token_test"
        result = get_hf_token("")
        self.assertEqual(result, "hf_env_token_test")

    def test_default_config_values(self):
        """Test default server configurations."""
        self.assertIsInstance(PORT, int)
        self.assertIsInstance(HOST, str)
        self.assertIsInstance(DEFAULT_MODEL, str)


class TestGraphTools(unittest.TestCase):

    def test_calculate_tool(self):
        """Test the calculate tool logic."""
        from backend.graph import calculate
        self.assertEqual(calculate.invoke("2 + 2"), "4")
        self.assertEqual(calculate.invoke("10 * 5"), "50")
        self.assertTrue("Error" in calculate.invoke("2 + import os"))


if __name__ == "__main__":
    unittest.main()
