import os
from typing import TYPE_CHECKING, Optional

from bovi_core.utils.dbfs_utils import with_dbutils

if TYPE_CHECKING:
    from pyspark.dbutils import DBUtils


class SecretsManager:
    def __init__(self, environment, config):
        self.environment = environment
        self.config = config  # self.config is the main Config instance

        self.scope_name = self._get_scope_name()

        if environment == "local":
            try:
                self.env_file = self.config.project.environments.local.env_file
            except AttributeError:
                self.env_file = ".env"

        # Validate scope name for Databricks environments
        if environment in ["databricks", "vscode_remote"] and not self.scope_name:
            raise ValueError(
                f"Secret scope name is required! Add 'secrets_scope' to "
                f"[tool.environments.{environment}] section in your TOML file."
            )

    def _get_scope_name(self):
        """Get scope name from config based on environment"""
        if self.environment in ["databricks", "vscode_remote"]:
            try:
                return self.config.project.environments.databricks.secrets_scope
            except AttributeError:
                # This will handle cases where the keys might be missing in the TOML
                return None
        return None

    def get_secret(self, key_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value using the key name"""
        if self.environment in ["databricks", "vscode_remote"]:
            return self._get_databricks_secret(key_name, default)
        else:
            return self._get_local_secret(key_name, default)

    @with_dbutils()
    def _get_databricks_secret(
        self, key_name: str, default: Optional[str] = None, dbutils: "DBUtils | None" = None
    ) -> Optional[str]:
        """Get secret from Databricks using scope and key name"""
        if dbutils is None:
            raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
        if self.scope_name is None:
            return default

        # Get the mapped secret name from the config
        mapped_key = None
        try:
            # Use raw secrets mapping to avoid circular dependency
            raw_mapping = getattr(self.config, "_raw_secrets_mapping", {})
            if raw_mapping and key_name in raw_mapping:
                mapped_key = raw_mapping[key_name]
                databricks_key = mapped_key
            else:
                databricks_key = key_name
        except (AttributeError, TypeError):
            databricks_key = key_name

        try:
            secret_value = dbutils.secrets.get(scope=self.scope_name, key=databricks_key)
            if self.config.experiment.verbose > 0:
                print(
                    f"✅ Retrieved secret '{key_name}' from scope '{self.scope_name}' "
                    f"(length: {len(secret_value) if secret_value else 0})"
                )
            return secret_value
        except Exception as e:
            if mapped_key:
                print(f"❌ Secret '{key_name}' not found!")
                print(
                    f"   📋 Mapping: '{key_name}' → '{mapped_key}' → Databricks secret '{databricks_key}'"
                )
                print(
                    f"   💡 Add to Databricks secrets scope '{self.scope_name}': {databricks_key}"
                )
            else:
                print(f"❌ Secret '{key_name}' not found!")
                print(
                    f"   📋 No mapping found in [tool.secrets]. Looking for Databricks secret '{databricks_key}'"
                )
                print("   💡 Either:")
                print(
                    f"      1. Add to Databricks secrets scope '{self.scope_name}': {databricks_key}"
                )
                print(
                    f'      2. Add mapping to pyproject.toml [tool.secrets]: {key_name} = "actual_secret_name"'
                )

            if default is not None:
                return default
            raise e

    def _get_local_secret(self, key_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from local environment"""
        # Get the mapped secret name from the config
        mapped_key = None
        env_key = None

        try:
            # Use raw secrets mapping to avoid circular dependency
            raw_mapping = getattr(self.config, "_raw_secrets_mapping", {})
            if raw_mapping and key_name in raw_mapping:
                mapped_key = raw_mapping[key_name]
                env_key = mapped_key.upper().replace("-", "_")
            else:
                # Fallback to direct key transformation if no mapping exists
                env_key = key_name.upper().replace("-", "_")
        except (AttributeError, TypeError):
            # Fallback to direct key transformation if config access fails
            env_key = key_name.upper().replace("-", "_")

        secret_value = os.environ.get(env_key, default)

        # Provide helpful error message if secret not found
        if secret_value is None and default is None:
            if mapped_key:
                print(f"❌ Secret '{key_name}' not found!")
                print(
                    f"   📋 Mapping: '{key_name}' → '{mapped_key}' → environment variable '{env_key}'"
                )
                print(f"   💡 Add to your .env file: {env_key}=your_secret_value_here")
            else:
                print(f"❌ Secret '{key_name}' not found!")
                print(
                    f"   📋 No mapping found in [tool.secrets]. Looking for environment variable '{env_key}'"
                )
                print("   💡 Either:")
                print(f"      1. Add to .env file: {env_key}=your_secret_value_here")
                print(
                    f'      2. Add mapping to pyproject.toml [tool.secrets]: {key_name} = "actual_secret_name"'
                )

        return secret_value

    def add_secret(
        self, key_name: str, secret_value: str, create_scope_if_needed: bool = True
    ) -> bool:
        """Add a secret to the secret scope (Databricks environments only)"""
        if self.environment not in ["databricks", "vscode_remote"]:
            print(
                f"⚠️ Adding secrets is only supported in Databricks environments. "
                f"Current environment: {self.environment}"
            )
            return False

        if not self.scope_name:
            print("❌ No secret scope configured. Cannot add secret.")
            return False

        from .utils.env_utils import add_secret_to_scope, create_secret_scope

        if create_scope_if_needed:
            if not create_secret_scope(self.scope_name):
                print(f"❌ Failed to create or verify scope '{self.scope_name}'")
                return False

        return add_secret_to_scope(self.scope_name, key_name, secret_value)

    def list_available_secrets(self) -> list:
        """List all secrets available in the configured scope"""
        if self.environment not in ["databricks", "vscode_remote"]:
            print(
                f"⚠️ Listing secrets is only supported in Databricks environments. "
                f"Current environment: {self.environment}"
            )
            return []

        if not self.scope_name:
            print("❌ No secret scope configured. Cannot list secrets.")
            return []

        print("🔍 Listing secrets in Databricks scope:", self.scope_name)
        from .utils.env_utils import list_secrets_in_scope

        return list_secrets_in_scope(self.scope_name)

    def populate_environment(self):
        """Load local .env file if needed"""
        if self.environment == "local":
            self._load_local_env()

    def _load_local_env(self):
        """Load .env file"""
        # Construct the full path to the .env file using the project_root from the config
        # This ensures it's found regardless of the current working directory.
        if hasattr(self, "env_file") and self.env_file:
            env_path = os.path.join(self.config.project.project_root, self.env_file)
            if os.path.exists(env_path):
                try:
                    from dotenv import load_dotenv

                    load_dotenv(env_path)
                    print(f"✅ Loaded environment variables from: {env_path}")
                except ImportError:
                    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
            else:
                print(f"⚠️  .env file not specified or not found at: {env_path}")
