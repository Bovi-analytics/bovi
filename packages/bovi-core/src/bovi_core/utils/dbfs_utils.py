import functools
import inspect
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.dbutils import DBUtils

import numpy as np


def repair_dbfs_path(path):
    if path.startswith("/dbfs/"):
        path = path.replace("/dbfs/", "dbfs:/")
    elif not path.startswith("dbfs:/"):
        path = f"dbfs:/{path.lstrip('/')}"
    return path


def with_dbutils():
    """Decorator to inject dbutils into functions that need it

    Usage:
        @with_dbutils()
        def my_function(param1, param2, dbutils=None):
            # dbutils will be automatically injected
    """

    def decorator(func):
        # Check if the function expects a dbutils parameter
        sig = inspect.signature(func)
        needs_dbutils = "dbutils" in sig.parameters

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If the function expects dbutils and it's not in kwargs, add it
            if needs_dbutils and "dbutils" not in kwargs:
                try:
                    from pyspark.dbutils import DBUtils
                    from pyspark.sql import SparkSession

                    spark = SparkSession.getActiveSession()
                    if spark:
                        kwargs["dbutils"] = DBUtils(spark)
                    else:
                        # Make the failure explicit
                        raise RuntimeError(
                            "Could not get active SparkSession to initialize dbutils. "
                            "This can happen if run on a Spark worker node or in a context "
                            "without Spark."
                        )
                except (ImportError, RuntimeError) as e:
                    # Re-raise with a clear message
                    raise RuntimeError(f"Failed to initialize DBUtils: {e}") from e
            return func(*args, **kwargs)

        return wrapper

    return decorator


def repair_path(path_param_name):
    """Decorator to repair DBFS paths - single responsibility

    Args:
        path_param_name: Name of the parameter that contains the path to repair
    """

    def decorator(func):
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find the path parameter and repair it
            if path_param_name in kwargs:
                # Path is in kwargs
                kwargs[path_param_name] = repair_dbfs_path(kwargs[path_param_name])
            elif len(args) > 0:
                # Find path parameter by name in positional args
                try:
                    path_index = param_names.index(path_param_name)
                    if path_index < len(args):
                        args = list(args)
                        args[path_index] = repair_dbfs_path(args[path_index])
                        args = tuple(args)
                except ValueError:
                    # Parameter not found, skip path repair
                    pass

            return func(*args, **kwargs)

        return wrapper

    return decorator


def remove_dbfs_prefix(path):
    path = repair_dbfs_path(path)
    return path[len("dbfs:/") :]


def handle_non_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Type not serializable")


@with_dbutils()
@repair_path("dir_path")
def dir_exists(dir_path, verbose=0, dbutils: "DBUtils | None" = None):
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    try:
        dbutils.fs.ls(dir_path)
        if verbose > 0:
            print(f"Directory exists: {dir_path}")
        return True
    except Exception:
        if verbose:
            print("Dir does not exist!")
        return False


@with_dbutils()
@repair_path("path")
def file_exists(path, verbose=0, dbutils: "DBUtils | None" = None):
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    try:
        if verbose:
            print(f"Looking for the following path: {path}")
        try:
            files_in_dir = dbutils.fs.ls(str(Path(path).parent))
        except Exception:
            if verbose:
                print(f"Directory does not exist: {str(Path(path).parent)}")
            return False
        if verbose:
            print(f"Directory exists: {str(Path(path).parent)}")
            print("Files in dir:")
            for i, f in enumerate(files_in_dir):
                print(f" File {i}:")
                print(f" - path: {f.path}")
                print(f" - name: {f.name}")
                print(f" - size: {f.size}")

        if any(path in file.path for file in files_in_dir):
            if verbose:
                print(f"File exists: {path}")
            return True
        if verbose:
            print(f"File does not exist: {path}")
        return False
    except Exception as e:
        print(f"Error listing file: {e}")
        raise e


@with_dbutils()
@repair_path("directory")
def list_of_files_with_substring(
    directory, substring, replace_dbfs_prefix=False, dbutils: "DBUtils | None" = None, verbose=0
):
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    try:
        # List all files in the directory
        if verbose:
            print(f"dir: {directory}")

        files = dbutils.fs.ls(directory)
        if verbose:
            print("all files in dir; showing first 5 and last 5:")
            print(*files[:5], sep="\n")
            print(*files[-5:], sep="\n")

        matched_files = [file.path for file in files if substring in file.path]
        if verbose:
            print("matched files; showing first 5 and last 5:")
            print(*matched_files[:5], sep="\n")
            print(*matched_files[-5:], sep="\n")

        if replace_dbfs_prefix:
            matched_files = [path.replace("dbfs:/", "/dbfs/") for path in matched_files]
            if verbose:
                print("files replaced; showing first 5 and last 5:")
                print(*matched_files[:5], sep="\n")
                print(*matched_files[-5:], sep="\n")
        return matched_files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


@with_dbutils()
@repair_path("path")
def get_file_size_dbfs(path, verbose=0, dbutils: "DBUtils | None" = None):
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    if not file_exists(path, verbose):
        return 0
    # Gets all the bytes from the file (max is 65536)
    return dbutils.fs.ls(path)[0].size


@with_dbutils()
@repair_path("path")
def create_dbfs_dir(path, exists_ok=True, verbose=0, dbutils: "DBUtils | None" = None) -> bool:
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    try:
        # Check if the directory exists
        try:
            if dir_exists(path, verbose=verbose):
                if exists_ok:
                    if verbose > 0:
                        print(f"Directory already exists: {path}")
                    return True
                else:
                    if verbose > 0:
                        print(f"Directory already exists and exists_ok=False: {path}")
                    return False
        except Exception as e:
            if "java.io.FileNotFoundException" in str(e):
                if verbose > 0:
                    print(f"Directory does not exist, will create: {path}")
            else:
                print(f"Error checking directory: {e}")
                return False

        # Create the directory
        if verbose > 0:
            print(f"Trying to create dir: {path}")

        try:
            dbutils.fs.mkdirs(path)
            if verbose > 0:
                print(f"Directory created: {path}")
        except Exception as e:
            print(f"Could not create dir: {e}")
            return False

        # Add a small delay to ensure filesystem updates
        time.sleep(0.5)

        # Verify if the directory now exists
        try:
            if dir_exists(path, verbose=verbose):
                if verbose > 0:
                    print(f"Directory verified: {path}")
                return True
            else:
                if verbose > 0:
                    print(f"Directory creation failed verification: {path}")
                return False
        except Exception as e:
            print(f"Error verifying directory: {e}")
            return False

    except Exception as e:
        print(f"Error creating directory: {e}")
        return False


@with_dbutils()
@repair_path("path")
def save_file_to_dbfs(
    path, data, overwrite=False, verbose=0, dbutils: "DBUtils | None" = None
) -> bool:
    if dbutils is None:
        raise RuntimeError("dbutils is required (ensure @with_dbutils applied)")
    # Check if the file exists; if so, handle according to the overwrite flag
    if file_exists(path, verbose) and not overwrite:
        if verbose > 0:
            print(f"File already exists in dbfs: {path}")
        return path

    # Ensure the parent directory exists (create it if missing)
    parent_dir = str(Path(path).parent)
    if not dir_exists(parent_dir, verbose):
        if verbose > 0:
            print(f"Parent directory {parent_dir} does not exist. Creating it.")
        success = create_dbfs_dir(parent_dir, exists_ok=True, verbose=verbose)
        if not success:
            raise Exception(f"Could not create parent directory: {parent_dir}")
    # Attempt to save the file
    try:
        if verbose > 0:
            print(f"Saving file: {path}")

        # Write binary or string data directly to the file
        try:
            if verbose:
                print(f"Saving binary data to file: {path}")
            if isinstance(data, bytes):
                try:
                    if verbose:
                        print(f"Converting binary data to string for file: {path}")
                    data = data.decode("utf-8")
                except Exception:
                    raise Exception(f"Could not convert binary data to string for file: {path}")
            dbutils.fs.put(path, data, overwrite=True)
        except Exception:
            if verbose:
                print(f"Could not write to file: {path}")
            raise Exception(f"Could not write to file: {path}")

        if verbose > 0:
            print(f"File saved: {path}")
        return path
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


@repair_path("dbfs_path")
def load_file_from_dbfs_to_temp(dbfs_path, temp_path, verbose=0):
    try:
        # Confirm source file exists
        if not file_exists(dbfs_path):
            if verbose > 0:
                print(f"Source file not found at: {dbfs_path}")
            raise Exception(f"Source file not found at: {dbfs_path}")

        # Create a temporary directory in /local_disk0/
        temp_dir = str(Path(temp_path).parent)
        # Use os as we are creating it on the drive (the compute cluster)
        os.makedirs(temp_dir, exist_ok=True)

        # Create a unique filename based on the original filename
        original_filename = str(Path(dbfs_path).name)
        temp_file_path = os.path.join(temp_dir, original_filename)

        # Check if the file already exists in the temporary location
        if os.path.exists(temp_file_path):
            # Verify if its the same file by comparing file sizes
            if get_file_size_dbfs(dbfs_path) == os.path.getsize(temp_file_path):
                if verbose > 0:
                    print(f"File already exists in temporary location: {temp_file_path}")
                return temp_file_path
            else:
                if verbose > 0:
                    print("File exists but has different size. Will replace it.")

        # Copy the file from DBFS to local temp directory

        # OPTION 1: Using regular Python file operations
        # Convert dbfs:/ path to /dbfs/ format for standard Python operations
        # This is needed as we are now working with the compute cluster.
        # This means that dbfs:/ is not supported and we need to use /dbfs/ instead
        print(f"Updating dbfs_path to /dbfs/ for Python operations (old path {dbfs_path})")
        local_dbfs_path = dbfs_path.replace("dbfs:/", "/dbfs/")
        print(f"Updated dbfs_path is {local_dbfs_path}")
        print(f"Copying file from {local_dbfs_path} to {temp_file_path}")
        shutil.copy(local_dbfs_path, temp_file_path)

        # OPTION 2: Using dbutils.fs
        # Here we need to add file: protocol to indicate local file system destination
        # dbutils.fs.cp(dbfs_path, f"file:{temp_file_path}")

        if verbose > 0:
            print(f"Copied file from {dbfs_path} to temporary location: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        print(f"Error copying file to temporary location: {e}")
        return None


# UNTESTED
@repair_path("dbfs_path")
def copy_file_from_dbfs_to_local(dbfs_path, dest_dir, verbose=0):
    """
    Copy a file from DBFS to any local destination path on the compute node.

    Args:
        dbfs_path (str): Source path in DBFS (starts with 'dbfs:/')
        dest_dir (str): Destination directory on the local file system
        verbose (int): Verbosity level
        dbutils: Databricks utilities object (automatically provided by decorator)

    Returns:
        str: Path to the copied file or None if there was an error
    """
    try:
        # Confirm source file exists
        if not file_exists(dbfs_path):
            if verbose > 0:
                print(f"Source file not found at: {dbfs_path}")
            raise Exception(f"Source file not found at: {dbfs_path}")

        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        # Extract original filename from the DBFS path
        original_filename = str(Path(dbfs_path).name)
        dest_file_path = os.path.join(dest_dir, original_filename)

        # Check if the file already exists in the destination
        if os.path.exists(dest_file_path):
            # Verify if it's the same file by comparing file sizes
            if get_file_size_dbfs(dbfs_path) == os.path.getsize(dest_file_path):
                if verbose > 0:
                    print(f"File already exists at destination: {dest_file_path}")
                return dest_file_path
            else:
                if verbose > 0:
                    print("File exists but has different size. Will replace it.")

        # Convert dbfs:/ path to /dbfs/ format for standard Python file operations
        local_dbfs_path = dbfs_path.replace("dbfs:/", "/dbfs/")

        # Copy the file using standard Python file operations
        import shutil

        shutil.copy(local_dbfs_path, dest_file_path)

        if verbose > 0:
            print(f"Copied file from {dbfs_path} to: {dest_file_path}")
        return dest_file_path

    except Exception as e:
        print(f"Error copying file to destination: {e}")
        return None


# UNTESTED
def save_worker_image_to_dbfs(frame, filepath, verbose=0):
    """Save image to DBFS directly from worker node"""
    try:
        import os

        import cv2

        # Ensure filepath starts with /dbfs/
        if not filepath.startswith("/dbfs/"):
            if filepath.startswith("dbfs:/"):
                filepath = filepath.replace("dbfs:/", "/dbfs/")
            else:
                filepath = f"/dbfs/{filepath}"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the image directly to the /dbfs/ path
        if verbose > 0:
            print(f"[DEBUG] Saving to {filepath}")

        # If already an image, save directly
        if len(frame.shape) in [2, 3]:
            result = cv2.imwrite(filepath, frame)
            if verbose > 0:
                print(f"[DEBUG] cv2.imwrite result: {result}")
        else:
            # If not properly shaped (like a 1D array), encode and save
            if verbose > 0:
                print("[DEBUG] Reshaping required before saving")
            return False

        return os.path.exists(filepath)

    except Exception as e:
        if verbose > 0:
            print(f"[DEBUG] Error saving to DBFS: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
