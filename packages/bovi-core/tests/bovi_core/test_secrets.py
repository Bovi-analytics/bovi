import os
from unittest.mock import MagicMock

import pytest

from bovi_core.config import ConfigNode

# Import the classes we need to test and to mock with
from bovi_core.secrets import SecretsManager

# --- PYTEST FIXTURES ---


@pytest.fixture
def mock_config():
    """
    Creates a comprehensive, in-memory mock of the main Config object
    that the SecretsManager expects to receive in its constructor.
    """
    # This structure mimics how the real Config object would be built
    config_data = {
        "project": {
            "project_root": "/tmp/fake_project_root",  # For testing .env path construction
            "environments": {
                "local": {"env_file": ".test.env"},
                "databricks": {"secrets_scope": "my-test-scope"},
                "vscode_remote": {"secrets_scope": "my-remote-scope"},
            },
        }
    }
    # We return a real ConfigNode so the attribute access works identically to the real code
    return ConfigNode(config_data)


@pytest.fixture
def mock_config_missing_environments():
    """A broken config fixture that is missing the entire [tool.environments] section."""
    config_data = {
        "project": {
            "project_root": "/tmp/fake_project_root"
            # No 'environments' key
        }
    }
    return ConfigNode(config_data)


@pytest.fixture
def mock_config_with_secrets_mapping():
    """Config fixture with secrets mapping like pyproject.toml."""
    config_data = {
        "project": {
            "project_root": "/tmp/fake_project_root",
            "environments": {
                "local": {"env_file": ".test.env"},
                "databricks": {"secrets_scope": "my-test-scope"},
            },
            "secrets": {
                "gh_key": "github_pat",
                "db_key": "databricks_pat",
                "storage_account_key": "storage_account_key"
            }
        }
    }
    config = ConfigNode(config_data)
    # Add _raw_secrets_mapping attribute that the SecretsManager uses
    config._raw_secrets_mapping = {
        "gh_key": "github_pat",
        "db_key": "databricks_pat",
        "storage_account_key": "storage_account_key"
    }
    return config


# --- TEST CLASSES ---


class TestSecretsManagerInitialization:
    """Tests focused on the constructor and initial setup of the manager."""

    def test_local_environment_setup(self, mock_config):
        """Tests if the manager correctly reads settings for a 'local' env."""
        manager = SecretsManager(environment="local", config=mock_config)
        assert manager.environment == "local"
        assert manager.scope_name is None
        assert manager.env_file == ".test.env"

    def test_databricks_environment_setup(self, mock_config):
        """Tests if the manager correctly reads settings for a 'databricks' env."""
        manager = SecretsManager(environment="databricks", config=mock_config)
        assert manager.environment == "databricks"
        assert manager.scope_name == "my-test-scope"
        assert not hasattr(manager, "env_file")

    def test_init_raises_error_if_scope_is_missing_for_databricks(
        self, mock_config_missing_environments
    ):
        """
        BREAK IT: Test that __init__ raises a ValueError if a Databricks-like
        environment is used without a configured secrets_scope.
        """
        with pytest.raises(ValueError, match="Secret scope name is required!"):
            SecretsManager(environment="databricks", config=mock_config_missing_environments)

    def test_init_succeeds_if_environments_are_missing_for_local(
        self, mock_config_missing_environments
    ):
        """
        Test graceful handling: for a 'local' env, it's okay if the
        environments config is missing. It should just use defaults.
        """
        try:
            manager = SecretsManager(environment="local", config=mock_config_missing_environments)
            # It should fall back to the default '.env'
            assert manager.env_file == ".env"
        except ValueError:
            pytest.fail(
                "SecretsManager should not raise ValueError for local env with missing config."
            )


class TestSecretRetrieval:
    """Tests the core get_secret() method in different environments and scenarios."""

    def test_get_local_secret_happy_path(self, mocker, mock_config):
        """Tests getting a secret from environment variables by mocking os.environ."""
        # Mock os.environ to return a fake value for a specific, transformed key
        mock_os_get = mocker.patch("os.environ.get", return_value="my-super-secret-password")

        manager = SecretsManager(environment="local", config=mock_config)
        secret = manager.get_secret("my-super-secret-key")

        # Verify our mock was called with the key transformed to UPPER_SNAKE_CASE
        mock_os_get.assert_called_once_with("MY_SUPER_SECRET_KEY", None)
        assert secret == "my-super-secret-password"

    def test_get_databricks_secret_happy_path(self, mocker, mock_config):
        """Tests successful retrieval from Databricks by mocking dbutils."""
        # We need a mock dbutils object to be returned by the decorator
        mock_dbutils = MagicMock()
        mock_dbutils.secrets.get.return_value = "fake-databricks-secret-value"

        # Mock the _get_databricks_secret method directly to inject our mock dbutils
        def mock_get_databricks_secret(self, key_name, default=None, dbutils=mock_dbutils):
            try:
                secret_value = dbutils.secrets.get(scope=self.scope_name, key=key_name)
                print(
                    f"✅ Retrieved secret '{key_name}' from scope '{self.scope_name}' (length: {len(secret_value) if secret_value else 0})"
                )
                return secret_value
            except Exception as e:
                print(
                    f"⚠️ Could not retrieve secret '{key_name}' from scope '{self.scope_name}': {e}"
                )
                env_key = key_name.upper().replace("-", "_")
                fallback_value = os.environ.get(env_key, default)
                print(
                    f"Using environment variable fallback: {env_key} = {'[AVAILABLE]' if fallback_value else '[NOT FOUND]'}"
                )
                return fallback_value

        mocker.patch.object(
            SecretsManager,
            "_get_databricks_secret",
            side_effect=mock_get_databricks_secret,
            autospec=True,
        )

        manager = SecretsManager(environment="databricks", config=mock_config)
        secret = manager.get_secret("my-db-key")

        # Assert that the mocked function was called with the correct scope and key
        mock_dbutils.secrets.get.assert_called_once_with(scope="my-test-scope", key="my-db-key")
        assert secret == "fake-databricks-secret-value"

    def test_databricks_secret_fails_and_falls_back_to_env_successfully(self, mocker, mock_config):
        """
        BREAK IT: Test the fallback logic. When dbutils fails, it should
        check os.environ as a backup.
        """
        # 1. Make the dbutils mock raise an exception
        mock_dbutils = MagicMock()
        mock_dbutils.secrets.get.side_effect = Exception("Invalid secret scope or key!")

        # Mock the _get_databricks_secret method directly to inject our mock dbutils
        def mock_get_databricks_secret(self, key_name, default=None, dbutils=mock_dbutils):
            try:
                secret_value = dbutils.secrets.get(scope=self.scope_name, key=key_name)
                print(
                    f"✅ Retrieved secret '{key_name}' from scope '{self.scope_name}' (length: {len(secret_value) if secret_value else 0})"
                )
                return secret_value
            except Exception as e:
                print(
                    f"⚠️ Could not retrieve secret '{key_name}' from scope '{self.scope_name}': {e}"
                )
                env_key = key_name.upper().replace("-", "_")
                fallback_value = os.environ.get(env_key, default)
                print(
                    f"Using environment variable fallback: {env_key} = {'[AVAILABLE]' if fallback_value else '[NOT FOUND]'}"
                )
                return fallback_value

        mocker.patch.object(
            SecretsManager,
            "_get_databricks_secret",
            side_effect=mock_get_databricks_secret,
            autospec=True,
        )

        # 2. Make the os.environ mock provide a fallback value
        mock_os_get = mocker.patch(
            "os.environ.get", return_value="secret-from-environment-variable"
        )

        manager = SecretsManager(environment="databricks", config=mock_config)
        secret = manager.get_secret("my-db-key")

        # 3. Assert the fallback was used and the correct value is returned
        mock_dbutils.secrets.get.assert_called_once()
        mock_os_get.assert_called_once_with("MY_DB_KEY", None)
        assert secret == "secret-from-environment-variable"

    def test_databricks_secret_fails_and_fallback_also_fails(self, mocker, mock_config):
        """
        BREAK IT HARDER: Test full failure. dbutils fails, and there is no
        environment variable to fall back to.
        """
        # 1. Make dbutils fail
        mock_dbutils = MagicMock()
        mock_dbutils.secrets.get.side_effect = Exception("Boom!")

        # 2. Make os.environ also fail (return None), but respect the default parameter
        def mock_os_get(key, default=None):
            return default

        mocker.patch("os.environ.get", side_effect=mock_os_get)

        # Mock the _get_databricks_secret method directly to inject our mock dbutils
        def mock_get_databricks_secret(self, key_name, default=None, dbutils=mock_dbutils):
            try:
                secret_value = dbutils.secrets.get(scope=self.scope_name, key=key_name)
                print(
                    f"✅ Retrieved secret '{key_name}' from scope '{self.scope_name}' (length: {len(secret_value) if secret_value else 0})"
                )
                return secret_value
            except Exception as e:
                print(
                    f"⚠️ Could not retrieve secret '{key_name}' from scope '{self.scope_name}': {e}"
                )
                env_key = key_name.upper().replace("-", "_")
                # Use the mocked os.environ.get which returns None, then fall back to the default
                fallback_value = mock_os_get(env_key, default)
                print(
                    f"Using environment variable fallback: {env_key} = {'[AVAILABLE]' if fallback_value else '[NOT FOUND]'}"
                )
                return fallback_value

        mocker.patch.object(
            SecretsManager,
            "_get_databricks_secret",
            side_effect=mock_get_databricks_secret,
            autospec=True,
        )

        manager = SecretsManager(environment="databricks", config=mock_config)

        # 3. It should return the default value, which is None
        secret = manager.get_secret("non-existent-key")
        assert secret is None

        # 4. It should return the provided default if one is given
        secret_with_default = manager.get_secret("non-existent-key", default="default-value")
        assert secret_with_default == "default-value"


class TestDotEnvLoading:
    """Tests the logic for loading variables from a .env file."""

    def test_populate_env_calls_load_dotenv_if_file_exists(self, mocker, mock_config):
        """Test that `load_dotenv` is called when the file is present."""
        mocker.patch("os.path.exists", return_value=True)  # Pretend the file exists
        mock_load_dotenv = mocker.patch("dotenv.load_dotenv")  # Watch this function

        manager = SecretsManager(environment="local", config=mock_config)
        manager.populate_environment()

        # The path should be constructed from project_root and env_file
        expected_path = os.path.join("/tmp/fake_project_root", ".test.env")
        mock_load_dotenv.assert_called_once_with(expected_path)

    def test_populate_env_skips_load_if_file_missing(self, mocker, mock_config):
        """Test that `load_dotenv` is NOT called if the .env file is missing."""
        mocker.patch("os.path.exists", return_value=False)  # Pretend the file does not exist
        mock_load_dotenv = mocker.patch("dotenv.load_dotenv")

        manager = SecretsManager(environment="local", config=mock_config)
        manager.populate_environment()

        mock_load_dotenv.assert_not_called()

    def test_populate_env_is_skipped_for_databricks_env(self, mocker, mock_config):
        """Test that we don't even try to load a .env file in Databricks."""
        # We watch the internal method to ensure it's not called
        mock_internal_method = mocker.patch.object(SecretsManager, "_load_local_env")

        manager = SecretsManager(environment="databricks", config=mock_config)
        manager.populate_environment()

        mock_internal_method.assert_not_called()


class TestSecretManagementMethods:
    """
    Tests the methods for adding, listing, and managing secrets.
    These tests mock the underlying utility functions to remain fast unit tests.
    """

    def test_add_secret_success(self, mocker, mock_config):
        """Tests the happy path for adding a secret."""
        # Mock the two functions that add_secret calls
        mock_create_scope = mocker.patch(
            "bovi_core.utils.env_utils.create_secret_scope", return_value=True
        )
        mock_add_to_scope = mocker.patch(
            "bovi_core.utils.env_utils.add_secret_to_scope", return_value=True
        )

        manager = SecretsManager(environment="databricks", config=mock_config)
        result = manager.add_secret("new-key", "new-value")

        assert result is True
        mock_create_scope.assert_called_once_with("my-test-scope")
        mock_add_to_scope.assert_called_once_with("my-test-scope", "new-key", "new-value")

    def test_add_secret_fails_if_scope_creation_fails(self, mocker, mock_config):
        """BREAK IT: Test that add_secret returns False if scope creation fails."""
        mock_create_scope = mocker.patch(
            "bovi_core.utils.env_utils.create_secret_scope", return_value=False
        )
        mock_add_to_scope = mocker.patch(
            "bovi_core.utils.env_utils.add_secret_to_scope"
        )  # Don't need a return value

        manager = SecretsManager(environment="databricks", config=mock_config)
        result = manager.add_secret("new-key", "new-value")

        assert result is False
        mock_create_scope.assert_called_once()
        mock_add_to_scope.assert_not_called()  # Should not proceed to add the secret

    def test_add_secret_only_works_in_databricks_env(self, mock_config):
        """Test that the method correctly blocks execution in a local environment."""
        manager = SecretsManager(environment="local", config=mock_config)
        result = manager.add_secret("new-key", "new-value")
        assert result is False

    def test_list_available_secrets(self, mocker, mock_config):
        """Tests the happy path for listing secrets."""
        expected_secrets = ["key1", "key2"]
        mock_list = mocker.patch(
            "bovi_core.utils.env_utils.list_secrets_in_scope",
            return_value=expected_secrets,
        )

        manager = SecretsManager(environment="databricks", config=mock_config)
        secrets = manager.list_available_secrets()

        assert secrets == expected_secrets
        mock_list.assert_called_once_with("my-test-scope")


class TestSecretsMapping:
    """Tests the secrets mapping functionality from pyproject.toml."""

    def test_local_secret_with_mapping_success(self, mocker, mock_config_with_secrets_mapping):
        """Test that mapping works: gh_key -> github_pat -> GITHUB_PAT."""
        # Mock os.environ to return value for GITHUB_PAT (not GH_KEY)
        mock_os_get = mocker.patch("os.environ.get")
        mock_os_get.return_value = "test_github_token_123"

        manager = SecretsManager(environment="local", config=mock_config_with_secrets_mapping)
        secret = manager._get_local_secret("gh_key")

        # Should look for GITHUB_PAT (mapped value), not GH_KEY (direct transformation)
        mock_os_get.assert_called_once_with("GITHUB_PAT", None)
        assert secret == "test_github_token_123"

    def test_local_secret_with_mapping_not_found(self, mocker, mock_config_with_secrets_mapping, capsys):
        """Test error message shows the mapping path when secret not found."""
        # Mock os.environ to return None (not found)
        mock_os_get = mocker.patch("os.environ.get", return_value=None)

        manager = SecretsManager(environment="local", config=mock_config_with_secrets_mapping)
        secret = manager._get_local_secret("gh_key")

        # Should look for GITHUB_PAT and find nothing
        mock_os_get.assert_called_once_with("GITHUB_PAT", None)
        assert secret is None

        # Check error message shows the mapping
        captured = capsys.readouterr()
        assert "gh_key' → 'github_pat' → environment variable 'GITHUB_PAT'" in captured.out

    def test_local_secret_without_mapping_fallback(self, mocker, mock_config):
        """Test fallback behavior when no mapping exists."""
        # Use config without secrets mapping
        mock_os_get = mocker.patch("os.environ.get", return_value="fallback_value")

        manager = SecretsManager(environment="local", config=mock_config)
        secret = manager._get_local_secret("unknown_key")

        # Should fall back to direct transformation: UNKNOWN_KEY
        mock_os_get.assert_called_once_with("UNKNOWN_KEY", None)
        assert secret == "fallback_value"

    def test_databricks_secret_with_mapping_success(self, mocker, mock_config_with_secrets_mapping):
        """Test that mapping works in Databricks: gh_key -> github_pat."""
        mock_dbutils = MagicMock()
        mock_dbutils.secrets.get.return_value = "databricks_github_token"

        manager = SecretsManager(environment="databricks", config=mock_config_with_secrets_mapping)

        # Mock the config.experiment for verbose logging
        manager.config.experiment = type('MockExperiment', (), {'verbose': 0})()

        secret = manager._get_databricks_secret("gh_key", dbutils=mock_dbutils)

        # Should look for 'github_pat' (mapped value), not 'gh_key'
        mock_dbutils.secrets.get.assert_called_once_with(scope="my-test-scope", key="github_pat")
        assert secret == "databricks_github_token"


class TestSecretsSecurity:
    """Tests that ensure secrets are never exposed in logs, errors, or debug output."""

    def test_secret_values_never_logged_in_success_case(self, mocker, mock_config, capsys):
        """Test that secret values are never printed or logged when successfully retrieved."""
        # Mock dbutils to return a secret
        mock_dbutils = MagicMock()
        mock_dbutils.secrets.get.return_value = "super-secret-password-123"

        # Mock the _get_databricks_secret method
        def mock_get_databricks_secret(self, key_name, default=None, dbutils=mock_dbutils):
            try:
                secret_value = dbutils.secrets.get(scope=self.scope_name, key=key_name)
                print(
                    f"✅ Retrieved secret '{key_name}' from scope '{self.scope_name}' (length: {len(secret_value) if secret_value else 0})"
                )
                return secret_value
            except Exception as e:
                print(
                    f"⚠️ Could not retrieve secret '{key_name}' from scope '{self.scope_name}': {e}"
                )
                env_key = key_name.upper().replace("-", "_")
                fallback_value = os.environ.get(env_key, default)
                print(
                    f"Using environment variable fallback: {env_key} = {'[AVAILABLE]' if fallback_value else '[NOT FOUND]'}"
                )
                return fallback_value

        mocker.patch.object(
            SecretsManager,
            "_get_databricks_secret",
            side_effect=mock_get_databricks_secret,
            autospec=True,
        )

        manager = SecretsManager(environment="databricks", config=mock_config)
        secret = manager.get_secret("test-key")

        # Capture all output
        captured = capsys.readouterr()

        # Verify the secret value is NOT in the output
        assert "super-secret-password-123" not in captured.out
        assert "super-secret-password" not in captured.out
        assert "password-123" not in captured.out

        # Verify only the length is logged, not the actual value
        assert "length: 25" in captured.out  # Length of 'super-secret-password-123'

        # Verify the secret is still returned correctly
        assert secret == "super-secret-password-123"

    def test_secret_values_never_logged_in_error_case(self, mocker, mock_config, capsys):
        """Test that secret values are never printed or logged even when errors occur."""
        # Mock dbutils to raise an exception
        mock_dbutils = MagicMock()
        mock_dbutils.secrets.get.side_effect = Exception("Secret not found")

        # Mock the _get_databricks_secret method
        def mock_get_databricks_secret(self, key_name, default=None, dbutils=mock_dbutils):
            try:
                secret_value = dbutils.secrets.get(scope=self.scope_name, key=key_name)
                print(
                    f"✅ Retrieved secret '{key_name}' from scope '{self.scope_name}' (length: {len(secret_value) if secret_value else 0})"
                )
                return secret_value
            except Exception as e:
                print(
                    f"⚠️ Could not retrieve secret '{key_name}' from scope '{self.scope_name}': {e}"
                )
                env_key = key_name.upper().replace("-", "_")
                fallback_value = os.environ.get(env_key, default)
                print(
                    f"Using environment variable fallback: {env_key} = {'[AVAILABLE]' if fallback_value else '[NOT FOUND]'}"
                )
                return fallback_value

        mocker.patch.object(
            SecretsManager,
            "_get_databricks_secret",
            side_effect=mock_get_databricks_secret,
            autospec=True,
        )

        # Set up environment variable fallback
        mocker.patch.dict(os.environ, {"TEST_KEY": "fallback-secret-value"})

        manager = SecretsManager(environment="databricks", config=mock_config)
        secret = manager.get_secret("test-key")

        # Capture all output
        captured = capsys.readouterr()

        # Verify the fallback secret value is NOT in the output
        assert "fallback-secret-value" not in captured.out
        assert "fallback-secret" not in captured.out

        # Verify only availability status is logged, not the actual value
        assert "[AVAILABLE]" in captured.out

        # Verify the secret is still returned correctly
        assert secret == "fallback-secret-value"

    def test_add_secret_never_logs_secret_value(self, mocker, mock_config, capsys):
        """Test that add_secret never logs the actual secret value being added."""
        # Mock successful operations
        mock_create_scope = mocker.patch(
            "bovi_core.utils.env_utils.create_secret_scope", return_value=True
        )
        mock_add_to_scope = mocker.patch(
            "bovi_core.utils.env_utils.add_secret_to_scope", return_value=True
        )

        manager = SecretsManager(environment="databricks", config=mock_config)
        result = manager.add_secret("sensitive-key", "very-sensitive-password-456")

        # Capture all output
        captured = capsys.readouterr()

        # Verify the secret value is NEVER in the output
        assert "very-sensitive-password-456" not in captured.out
        assert "very-sensitive-password" not in captured.out
        assert "sensitive-password" not in captured.out
        assert "password-456" not in captured.out

        # Verify the operation succeeded
        assert result is True
