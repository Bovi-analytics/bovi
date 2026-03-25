# Put here functions that can be used by all packages. Something like:


# def read_project_name_from_toml(toml_path: Path) -> str:
#     """Read and return the project name from a pyproject.toml file"""
#     try:
#         with open(toml_path, "rb") as f:
#             data = tomllib.load(f)
#         return data.get("project", {}).get("name", "unknown")
#     except (ImportError, FileNotFoundError, KeyError, ValueError):
#         return "unknown"
