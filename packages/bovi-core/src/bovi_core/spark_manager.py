"""
Spark Session Manager - Clean singleton pattern for managing Spark and DBUtils instances
"""

import threading
from typing import Optional

from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


class SparkManager:
    """Singleton manager for Spark session and DBUtils with thread safety"""

    _instance: Optional["SparkManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._spark: Optional[SparkSession] = None
            self._dbutils: Optional[DBUtils] = None
            self._workspace_url: Optional[str] = None
            self._initialized = True

    @property
    def spark(self) -> SparkSession:
        """Get or create Spark session"""
        if self._spark is None:
            self._spark = SparkSession.builder.getOrCreate()
        return self._spark

    @property
    def dbutils(self) -> DBUtils:
        """Get or create DBUtils instance"""
        if self._dbutils is None:
            self._dbutils = DBUtils(self.spark)
        return self._dbutils

    @property
    def workspace_url(self) -> Optional[str]:
        """Get workspace URL if available"""
        if self._workspace_url is None:
            try:
                self._workspace_url = self.spark.conf.get("spark.databricks.workspaceUrl")
            except Exception:
                self._workspace_url = None
        return self._workspace_url

    def reset(self):
        """Reset the manager (useful for testing)"""
        self._spark = None
        self._dbutils = None
        self._workspace_url = None


# Global instance - singleton access
_manager = SparkManager()


def get_spark() -> SparkSession:
    """Get the Spark session"""
    return _manager.spark


def get_dbutils() -> DBUtils:
    """Get the DBUtils instance"""
    return _manager.dbutils


def get_workspace_url() -> Optional[str]:
    """Get the workspace URL"""
    return _manager.workspace_url


def reset_spark_manager():
    """Reset the Spark manager (useful for testing)"""
    _manager.reset()
