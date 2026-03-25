"""
Tests for Dataset signature generation methods.

Tests cover:
- get_input_example() with various configurations
- _batch_samples() with different data types
- get_mlflow_signature() with and without model predictions
- get_signature_info() for debugging
"""

import io
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from bovi_core.ml.dataloaders.base import Dataset
from bovi_core.ml.dataloaders import ImageDataset


class MockImageSource:
    """Mock image source for testing"""

    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self):
        return self.size

    def load_item(self, key):
        """Return fake image data (PIL format)"""
        from PIL import Image

        # Create a simple RGB image
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def get_metadata(self, key):
        return {"label": key % 3, "path": f"image_{key}.jpg"}

    def get_keys(self):
        return list(range(self.size))


@pytest.fixture
def image_dataset():
    """Create a simple image dataset for testing"""
    source = MockImageSource(size=10)
    return ImageDataset(source)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with simple data"""

    class SimpleDataset(Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, index):
            return {
                "data": np.random.rand(3, 224, 224).astype(np.float32),
                "label": index % 3,
                "index": index,
            }

    return SimpleDataset(source=Mock())


class TestGetInputExample:
    """Tests for Dataset.get_input_example()"""

    def test_single_unbatched_sample(self, mock_dataset):
        """Test getting single unbatched sample"""
        example = mock_dataset.get_input_example(n_samples=1, batch=False)

        assert isinstance(example, dict)
        assert "data" in example
        assert "label" in example
        assert example["data"].shape == (3, 224, 224)

    def test_batched_samples(self, mock_dataset):
        """Test getting batched samples"""
        example = mock_dataset.get_input_example(n_samples=5, batch=True)

        assert isinstance(example, dict)
        assert "data" in example
        # First dimension should be batch size
        assert example["data"].shape[0] == 5
        assert example["data"].shape[1:] == (3, 224, 224)
        # Label should be array
        assert isinstance(example["label"], np.ndarray)
        assert example["label"].shape[0] == 5

    def test_unbatched_multiple_samples(self, mock_dataset):
        """Test getting multiple unbatched samples"""
        examples = mock_dataset.get_input_example(n_samples=3, batch=False)

        assert isinstance(examples, list)
        assert len(examples) == 3
        for ex in examples:
            assert isinstance(ex, dict)
            assert ex["data"].shape == (3, 224, 224)

    def test_custom_indices(self, mock_dataset):
        """Test with custom indices"""
        indices = [5, 10, 15]
        example = mock_dataset.get_input_example(
            n_samples=3,
            indices=indices,
            batch=True
        )

        assert example["index"].shape[0] == 3
        np.testing.assert_array_equal(example["index"], np.array(indices))

    def test_n_samples_exceeds_dataset(self, mock_dataset):
        """Test when n_samples exceeds dataset size"""
        # Dataset has 20 samples, request 100
        example = mock_dataset.get_input_example(n_samples=100, batch=True)

        # Should only return 20
        assert example["data"].shape[0] == 20

    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises ValueError"""

        class EmptyDataset(Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, index):
                return {}

        dataset = EmptyDataset(source=Mock())

        with pytest.raises(ValueError, match="Dataset is empty"):
            dataset.get_input_example()

    def test_invalid_indices_raises_error(self, mock_dataset):
        """Test that invalid indices raise ValueError"""
        invalid_indices = [5, 100, 200]  # 100, 200 out of range

        with pytest.raises(ValueError, match="Index out of range"):
            mock_dataset.get_input_example(
                n_samples=3,
                indices=invalid_indices,
                batch=True
            )


class TestBatchSamples:
    """Tests for Dataset._batch_samples()"""

    def test_batch_numpy_arrays(self, mock_dataset):
        """Test batching numpy arrays"""
        samples = [
            {"array": np.zeros((3, 4)), "id": i}
            for i in range(3)
        ]

        batched = mock_dataset._batch_samples(samples)

        assert batched["array"].shape == (3, 3, 4)
        np.testing.assert_array_equal(
            batched["array"],
            np.zeros((3, 3, 4))
        )

    def test_batch_scalars(self, mock_dataset):
        """Test batching scalar values"""
        samples = [
            {"value": i, "float_val": float(i) * 0.5}
            for i in range(3)
        ]

        batched = mock_dataset._batch_samples(samples)

        assert isinstance(batched["value"], np.ndarray)
        np.testing.assert_array_equal(batched["value"], np.array([0, 1, 2]))
        np.testing.assert_array_almost_equal(
            batched["float_val"],
            np.array([0.0, 0.5, 1.0])
        )

    def test_batch_strings(self, mock_dataset):
        """Test batching strings (keeps as list)"""
        samples = [
            {"name": f"sample_{i}"}
            for i in range(3)
        ]

        batched = mock_dataset._batch_samples(samples)

        assert isinstance(batched["name"], list)
        assert len(batched["name"]) == 3
        assert batched["name"] == ["sample_0", "sample_1", "sample_2"]

    def test_batch_mixed_types(self, mock_dataset):
        """Test batching mixed data types"""
        samples = [
            {
                "array": np.ones((2, 3)) * i,
                "label": i,
                "name": f"sample_{i}",
            }
            for i in range(3)
        ]

        batched = mock_dataset._batch_samples(samples)

        assert batched["array"].shape == (3, 2, 3)
        assert batched["label"].shape == (3,)
        assert batched["name"] == ["sample_0", "sample_1", "sample_2"]

    def test_batch_with_none_values(self, mock_dataset):
        """Test batching with None values"""
        samples = [
            {"data": np.ones((2,)), "optional": None if i % 2 == 0 else i}
            for i in range(3)
        ]

        batched = mock_dataset._batch_samples(samples)

        assert batched["data"].shape == (3, 2)
        # Optional field with mixed None and int values becomes object array
        assert batched["optional"].dtype == object

    def test_batch_all_none(self, mock_dataset):
        """Test batching all None values"""
        samples = [
            {"field": None}
            for _ in range(3)
        ]

        batched = mock_dataset._batch_samples(samples)

        assert batched["field"] is None

    def test_batch_empty_list(self, mock_dataset):
        """Test batching empty list"""
        batched = mock_dataset._batch_samples([])
        assert batched == {}


class TestGetMLflowSignature:
    """Tests for Dataset.get_mlflow_signature()"""

    @pytest.mark.skipif(
        pytest.importorskip("mlflow", minversion=None) is None,
        reason="mlflow not installed"
    )
    def test_input_only_signature(self, mock_dataset):
        """Test generating input-only signature"""
        signature = mock_dataset.get_mlflow_signature(model=None, n_samples=5)

        assert signature is not None
        assert signature.inputs is not None
        # Should have input schema
        assert hasattr(signature.inputs, 'to_dict')

    @pytest.mark.skipif(
        pytest.importorskip("mlflow", minversion=None) is None,
        reason="mlflow not installed"
    )
    def test_signature_with_model(self, mock_dataset):
        """Test generating signature with model predictions"""
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(5, 10)

        signature = mock_dataset.get_mlflow_signature(
            model=mock_model,
            n_samples=5
        )

        assert signature is not None
        assert signature.inputs is not None
        # Model should have been called
        mock_model.predict.assert_called_once()

    def test_signature_missing_mlflow(self, mock_dataset):
        """Test error when mlflow not installed"""
        with patch('mlflow.models.infer_signature',
                   side_effect=ImportError("mlflow not found")):
            # Should raise ImportError
            with pytest.raises(ImportError):
                mock_dataset.get_mlflow_signature()


class TestGetSignatureInfo:
    """Tests for Dataset.get_signature_info()"""

    def test_signature_info_structure(self, mock_dataset):
        """Test structure of signature info"""
        info = mock_dataset.get_signature_info()

        assert "num_samples" in info
        assert "sample_fields" in info
        assert "field_types" in info
        assert "field_shapes" in info
        assert "example_sample" in info

    def test_signature_info_values(self, mock_dataset):
        """Test values in signature info"""
        info = mock_dataset.get_signature_info()

        assert info["num_samples"] == 20
        assert set(info["sample_fields"]) == {"data", "label", "index"}
        assert info["field_types"]["data"] == "ndarray"
        assert info["field_shapes"]["data"] == (3, 224, 224)
        assert info["field_shapes"]["label"] == ()

    def test_signature_info_with_image_dataset(self, image_dataset):
        """Test signature info with real ImageDataset"""
        info = image_dataset.get_signature_info()

        assert info["num_samples"] == 10
        assert "image" in info["sample_fields"]
        assert "label" in info["sample_fields"]
        # Image might be PIL Image or ndarray depending on transforms
        assert isinstance(info["field_types"]["image"], str)


class TestIntegration:
    """Integration tests combining multiple methods"""

    def test_example_to_signature_flow(self, mock_dataset):
        """Test complete flow from example to signature"""
        # Get input example
        example = mock_dataset.get_input_example(n_samples=5, batch=True)
        assert example["data"].shape[0] == 5

        # Get signature info
        info = mock_dataset.get_signature_info()
        assert info["num_samples"] == 20

    def test_with_image_dataset(self, image_dataset):
        """Integration test with real ImageDataset"""
        # Get signature info
        info = image_dataset.get_signature_info()
        assert info["num_samples"] == 10

        # Get input example
        example = image_dataset.get_input_example(n_samples=3, batch=True)
        assert "image" in example
        assert "label" in example

        # Check types (images might be list if they can't be stacked)
        assert hasattr(example["image"], '__len__')
        # If batched into array, check shape
        if hasattr(example["image"], 'shape'):
            assert example["image"].shape[0] == 3  # batch size
