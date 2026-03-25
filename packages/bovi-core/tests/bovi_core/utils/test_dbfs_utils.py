from unittest.mock import MagicMock, call, patch

# Import the functions to be tested
from bovi_core.utils.dbfs_utils import (
    create_dbfs_dir,
    dir_exists,
    file_exists,
    get_file_size_dbfs,
    list_of_files_with_substring,
    repair_dbfs_path,
    save_file_to_dbfs,
)

# --- Test Classes ---


class TestDbfsPathHandling:
    """Tests the DBFS path normalization logic."""

    def test_repair_dbfs_path(self):
        """Ensures that paths are correctly formatted for DBFS."""
        assert repair_dbfs_path("/dbfs/some/path") == "dbfs:/some/path"
        assert repair_dbfs_path("dbfs:/some/path") == "dbfs:/some/path"
        assert repair_dbfs_path("some/path") == "dbfs:/some/path"
        assert repair_dbfs_path("/other/path") == "dbfs:/other/path"


class TestDbfsExistenceChecks:
    """Tests dir_exists and file_exists functions."""

    def test_dir_exists_happy_path(self, mock_dbutils):
        """Test dir_exists when the directory is present."""
        mock_dbutils.fs.ls.return_value = [MagicMock()]  # Simulate non-empty list
        assert dir_exists("dbfs:/my/dir", dbutils=mock_dbutils) is True
        mock_dbutils.fs.ls.assert_called_once_with("dbfs:/my/dir")

    def test_dir_exists_failure_path(self, mock_dbutils):
        """Test dir_exists when the directory is missing, causing an exception."""
        mock_dbutils.fs.ls.side_effect = Exception("File not found")
        assert dir_exists("dbfs:/my/dir", dbutils=mock_dbutils) is False

    def test_file_exists_happy_path(self, mock_dbutils):
        """Test file_exists when the file is present in the parent directory listing."""
        file_path = "dbfs:/my/dir/file.txt"
        parent_dir = "dbfs:/my/dir"

        mock_file_info = MagicMock()
        mock_file_info.path = file_path
        mock_dbutils.fs.ls.return_value = [mock_file_info]

        assert file_exists(file_path, dbutils=mock_dbutils) is True
        mock_dbutils.fs.ls.assert_called_once_with(parent_dir)

    def test_file_exists_failure_path(self, mock_dbutils):
        """Test file_exists when the file is not in the parent directory listing."""
        file_path = "dbfs:/my/dir/file.txt"
        mock_dbutils.fs.ls.return_value = []
        assert file_exists(file_path, dbutils=mock_dbutils) is False


class TestDbfsFileOperations:
    """Tests functions that create or modify files and directories on DBFS."""

    @patch("bovi_core.utils.dbfs_utils.dir_exists")
    def test_create_dbfs_dir_success(self, mock_dir_exists, mock_dbutils):
        """Test successful creation of a directory."""
        path = "dbfs:/new/dir"
        mock_dir_exists.side_effect = [False, True]

        assert create_dbfs_dir(path, dbutils=mock_dbutils) is True

        mock_dir_exists.assert_has_calls([call(path, verbose=0), call(path, verbose=0)])
        mock_dbutils.fs.mkdirs.assert_called_once_with(path)

    @patch("bovi_core.utils.dbfs_utils.dir_exists")
    def test_create_dbfs_dir_already_exists(self, mock_dir_exists, mock_dbutils):
        """Test that it returns True and does nothing if the directory exists."""
        path = "dbfs:/existing/dir"
        mock_dir_exists.return_value = True

        assert create_dbfs_dir(path, dbutils=mock_dbutils) is True

        mock_dir_exists.assert_called_once_with(path, verbose=0)
        mock_dbutils.fs.mkdirs.assert_not_called()

    @patch("bovi_core.utils.dbfs_utils.dir_exists", return_value=False)
    @patch("bovi_core.utils.dbfs_utils.create_dbfs_dir", return_value=True)
    def test_save_file_to_dbfs_creates_parent_and_saves(
        self, mock_create_dir, mock_dir_exists, mock_dbutils
    ):
        file_path = "dbfs:/new/parent/file.txt"
        parent_dir = "dbfs:/new/parent"
        file_data = "Hello, world!"

        with patch("bovi_core.utils.dbfs_utils.file_exists", return_value=False):
            save_file_to_dbfs(file_path, file_data, dbutils=mock_dbutils)

        mock_dir_exists.assert_called_once_with(parent_dir, 0)
        mock_create_dir.assert_called_once_with(parent_dir, exists_ok=True, verbose=0)
        mock_dbutils.fs.put.assert_called_once_with(file_path, file_data, overwrite=True)

    def test_list_of_files_with_substring(self, mock_dbutils):
        """Test that file listing and filtering works correctly."""
        dir_path = "dbfs:/data"
        mock_file1 = MagicMock()
        mock_file1.path = "dbfs:/data/image_01.jpg"
        mock_file2 = MagicMock()
        mock_file2.path = "dbfs:/data/image_02.jpg"
        mock_file3 = MagicMock()
        mock_file3.path = "dbfs:/data/metadata.json"

        mock_dbutils.fs.ls.return_value = [mock_file1, mock_file2, mock_file3]

        result = list_of_files_with_substring(dir_path, ".jpg", dbutils=mock_dbutils)

        assert result == ["dbfs:/data/image_01.jpg", "dbfs:/data/image_02.jpg"]
        mock_dbutils.fs.ls.assert_called_once_with(dir_path)

    def test_get_file_size_dbfs(self, mock_dbutils):
        """Test retrieval of file size."""
        file_path = "dbfs:/my/file.txt"

        with patch("bovi_core.utils.dbfs_utils.file_exists", return_value=True):
            mock_file_info = MagicMock()
            mock_file_info.size = 1024
            mock_dbutils.fs.ls.return_value = [mock_file_info]

            size = get_file_size_dbfs(file_path, dbutils=mock_dbutils)
            assert size == 1024
            mock_dbutils.fs.ls.assert_called_once_with(file_path)
