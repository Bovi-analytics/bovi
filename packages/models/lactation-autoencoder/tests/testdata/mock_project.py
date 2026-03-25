MOCK_PYPROJECT_TOML_ELABORATE = """
[project]
name = "elaborate_test_project"
version = "1.0.0"
requires-python = ">=3.11, <3.12"

[tool.secrets]
my_secret_key = "actual-secret-name-in-vault"

[tool.databricks]
cluster_name = "elaborate_test_cluster"

[tool.blob_storage]
storage_account_name = "teststorage"
container_name = "testcontainer"

[tool.environments.local]
env_file = ".env"

[tool.environments.databricks]
secrets_scope = "my-test-scope"

[tool.environments.vscode_remote]
secrets_scope = "my-remote-scope"
"""
