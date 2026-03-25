"""Tests for DataSource ABC."""

from bovi_core.ml.dataloaders.base import DataSource


class MockDataSource(DataSource):
    """Mock implementation for testing"""

    def __init__(self):
        self.data = {0: b"data0", 1: b"data1"}

    def __len__(self):
        return len(self.data)

    def load_item(self, key):
        return self.data[key]

    def get_metadata(self, key):
        return {"key": key, "size": len(self.data[key])}

    def get_keys(self):
        return list(self.data.keys())


def test_data_source_interface():
    """Test DataSource interface"""
    source = MockDataSource()
    assert len(source) == 2
    assert source.load_item(0) == b"data0"
    assert source.get_metadata(0)["size"] == 5
    assert source.get_keys() == [0, 1]


def test_data_source_context_manager():
    """Test context manager support"""
    with MockDataSource() as source:
        assert len(source) == 2
    # Should not raise after context exit
