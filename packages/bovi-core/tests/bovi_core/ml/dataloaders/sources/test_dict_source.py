"""Tests for DictSource."""

from bovi_core.ml.dataloaders.sources.dict_source import DictSource


class TestDictSource:
    def test_single_item(self):
        source = DictSource([{"milk": [1.0, 2.0], "parity": 1}])
        assert len(source) == 1
        assert source.load_item(0)["parity"] == 1

    def test_multiple_items(self):
        source = DictSource([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(source) == 3
        assert source.load_item(2) == {"a": 3}

    def test_get_keys(self):
        source = DictSource([{"x": 1}, {"x": 2}])
        assert source.get_keys() == [0, 1]

    def test_iteration(self):
        data = [{"a": 1}, {"a": 2}]
        assert list(DictSource(data)) == data

    def test_empty(self):
        assert len(DictSource([])) == 0
