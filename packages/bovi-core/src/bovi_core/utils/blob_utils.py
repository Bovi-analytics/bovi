import json
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np

# TYPE_CHECKING is True during static type checking but False at runtime.
# This allows us to import DBUtils for type annotations without causing
# ImportError when pyspark is not available (e.g., in local development).
if TYPE_CHECKING:
    from pyspark.dbutils import DBUtils


import bovi_core.utils.dbfs_utils as dbfs_utils
from bovi_core.config import Config, with_config


@dbfs_utils.with_dbutils()
@with_config
def mount_blob_container_from_config(
    config: Config,
    dbutils: "DBUtils | None" = None,
    mount_point_override: Optional[str] = None,
    verbose: int = 0,
):
    """
    Mounts the blob container specified in the config to a DBFS mount point.
    It uses the 'mount_point' from the [tool.blob_storage] section of the TOML file.
    This can be optionally overridden by the 'mount_point_override' argument.
    """
    # Check if we're running on Databricks
    if config.environment != "databricks":
        if verbose > 0:
            print(
                f"⚠️  Not running on Databricks (environment: {config.environment}), skipping mount"
            )
        return False

    # Type assertion: dbutils should be available in Databricks environment
    assert dbutils is not None, "dbutils should be injected by @with_dbutils decorator"

    # Determine the mount point: use override if provided, otherwise use config
    try:
        mount_point_from_config = config.project.blob_storage.mount_point
    except AttributeError:
        mount_point_from_config = None

    final_mount_point = mount_point_override or mount_point_from_config

    if not final_mount_point:
        raise ValueError(
            "Mount point not found. Specify it in your pyproject.toml under "
            "[tool.blob_storage.mount_point] or provide a 'mount_point_override' argument."
        )

    if verbose > 0:
        print(f"Attempting to mount to: {final_mount_point}")

    # Check if already mounted
    if any(mount.mountPoint == final_mount_point for mount in dbutils.fs.mounts()):
        if verbose > 0:
            print(f"Mount point {final_mount_point} is already mounted.")
        return True

    # If not mounted, proceed with mounting
    try:
        storage_account_name = config.project.blob_storage.storage_account_name
        container_name = config.project.blob_storage.container_name
        storage_account_key = config.secrets.storage_account_key
    except AttributeError as e:
        raise ValueError(f"Missing required blob storage configuration: {e}")

    source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net"
    extra_configs = {
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_key
    }

    if verbose > 0:
        print(f"Source: {source}")
        print(f"Mount Point: {final_mount_point}")

    try:
        mounted = dbutils.fs.mount(
            source=source,
            mount_point=final_mount_point,
            extra_configs=extra_configs,
        )
        if mounted:
            if verbose > 0:
                print(
                    f"✅ Successfully mounted blob container '{container_name}' at "
                    f"{final_mount_point}"
                )
        return mounted
    except Exception as e:
        print(f"❌ Could not mount blob container '{container_name}' at {final_mount_point}: {e}")
        return False


@with_config
def list_blob(blob_path: str, config: Config | None = None, verbose: int = 0) -> List[str]:
    """
    Lists blobs starting with a specific path prefix.

    Args:
        blob_path (str): The path/prefix to the blob to list.
        config (Config): The configuration object. Auto-provided by @with_config decorator.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        List[str]: A list of blob names matching the path prefix. Returns an empty list
            if no blobs are found.
    """
    if not config:
        raise AttributeError("Config not passed or retrieved with decorator.")
    try:
        container_client = config.container_client
        blob_list = container_client.list_blobs(name_starts_with=blob_path)
        blob_names = [blob.name for blob in blob_list]
        if verbose > 0:
            if blob_names:
                print(f"Found {len(blob_names)} blob(s) under path: {blob_path}")
            else:
                print(f"No blobs found at path: {blob_path}")
        return blob_names
    except Exception as e:
        print(f"Could not retrieve blob(s) at path {blob_path}: {e}")
        raise e


@with_config
def list_blobs_by_pattern(
    dir_path: str, substring: str, config: Config | None = None, verbose: int = 0
) -> List[str]:
    """
    Lists blobs in a directory that contain a specific substring.

    Args:
        dir_path (str): The directory path within the container.
        substring (str): The substring to match in the blob name.
        config (Config): The configuration object. Auto-provided by @with_config decorator.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        List[str]: A list of matching blob names.
    """
    if not config:
        raise AttributeError("Config not passed or retrieved with decorator.")
    container_client = config.container_client
    if not dir_path.endswith("/"):
        dir_path += "/"
    blob_list = container_client.list_blobs(name_starts_with=dir_path)
    if verbose > 0:
        unfiltered_names = [blob.name for blob in blob_list]
        print(f"Unfiltered blobs: {unfiltered_names}")
        # Refetch since the generator was consumed
        blob_list = container_client.list_blobs(name_starts_with=dir_path)

    filtered_blobs = [
        blob.name for blob in blob_list if substring is None or substring in blob.name
    ]
    if verbose > 0:
        print(f"Filtered blobs: {filtered_blobs}")
    return filtered_blobs


@with_config
def get_file_blob(blob_path: str, config: Config, verbose: int = 0) -> bytes:
    """Downloads a blob from storage as bytes."""
    container_client = config.container_client
    try:
        blob_client = container_client.get_blob_client(blob=blob_path)
        if verbose > 0:
            print(f"Retrieved blob client for {blob_path}")
        stream = blob_client.download_blob().readall()
        if verbose > 0:
            print(f"Downloaded blob {blob_path}")
        return stream
    except Exception as e:
        print(f"Failed to download blob {blob_path}: {e}")
        raise e


@with_config
def get_image_stream(blob_path: str, config: Config, mode=cv2.IMREAD_COLOR, verbose: int = 0):
    """Downloads a blob and decodes it as an image."""
    stream = get_file_blob(blob_path, config=config, verbose=verbose)
    np_img_data = np.frombuffer(stream, np.uint8)
    try:
        image = cv2.imdecode(np_img_data, mode)
        if verbose > 0:
            print(f"Decoded image from {blob_path}")
        return image
    except Exception as e:
        print(f"Failed to decode image from blob {blob_path}: {e}")
        raise e


@with_config
def get_json_stream(blob_path: str, config: Config) -> dict:
    """Downloads and parses a JSON blob."""
    stream = get_file_blob(blob_path, config=config)
    json_data = json.loads(stream)
    return json_data


@with_config
def save_image_to_blob(image, blob_path: str, config: Config, verbose: int = 0):
    """Encodes an image and uploads it to a blob."""
    container_client = config.container_client
    _, img_encoded = cv2.imencode(".png", image)
    blob_client = container_client.get_blob_client(blob=blob_path)
    blob_client.upload_blob(img_encoded.tobytes(), overwrite=True)
    if verbose > 0:
        print(f"Saved image to {blob_path}")


def handle_non_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Type not serializable")


@with_config
def save_file_to_blob(data, blob_path: str, file_extension: str, config: Config, verbose: int = 0):
    """Saves data to a blob, encoding it based on the file extension."""
    container_client = config.container_client
    blob_client = container_client.get_blob_client(blob=blob_path)

    if file_extension == "json":
        file_data = json.dumps(data, indent=4, default=handle_non_serializable).encode("utf-8")
    elif file_extension in ["index", "data", "events", "mp4"]:
        if not isinstance(data, bytes):
            raise ValueError("Data must be in bytes format for this extension.")
        file_data = data
    elif file_extension in ["txt", "yaml"]:
        if not isinstance(data, str):
            raise ValueError("Data must be in string format for this extension.")
        file_data = data.encode("utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    blob_client.upload_blob(file_data, overwrite=True)
    if verbose > 0:
        print(f"Saved data to {blob_path} as a {file_extension} file.")


@with_config
def save_file_to_dbfs_from_blob(file_path: str, config: Config, verbose: int = 0):
    """Downloads a file from blob storage and saves it to DBFS if it doesn't already exist."""
    file_path_dbfs = dbfs_utils.repair_dbfs_path(file_path)
    if dbfs_utils.file_exists(file_path_dbfs, verbose=verbose):
        if verbose > 0:
            print("File already exists in DBFS, skipping download from blob.")
        return file_path_dbfs
    else:
        if verbose > 0:
            print(f"Downloading file {file_path} to DBFS from blob storage.")
        # Check if blob exists before trying to download
        if list_blob(file_path, config=config, verbose=verbose):
            file_bytes = get_file_blob(file_path, config=config, verbose=verbose)
            if dbfs_utils.save_file_to_dbfs(
                file_path, file_bytes, overwrite=False, verbose=verbose
            ):
                return file_path_dbfs
    return False
