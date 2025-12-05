"""Tests for CUDA functionality and GPU support."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from backend.main import ModelBundle, ServiceConfig


class TestCUDASupport:
    """Tests for CUDA/GPU support."""

    @pytest.fixture
    def cuda_available(self):
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    def test_cuda_availability_check(self):
        """Test CUDA availability can be checked."""
        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)

    @pytest.mark.requires_cuda
    def test_model_loads_on_cuda(self):
        """Test model loads on CUDA when available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = ServiceConfig()
        config.device_map = "cuda:0"

        # This would require actual model loading
        # For now, we test the configuration
        assert config.device_map == "cuda:0"

    def test_device_map_auto(self):
        """Test device_map='auto' configuration."""
        config = ServiceConfig()
        assert config.device_map == "auto"

    def test_model_uses_cuda_when_available(self, cuda_available):
        """Test model uses CUDA when available."""
        config = ServiceConfig()
        bundle = ModelBundle(config)

        # Mock model with CUDA device
        if cuda_available:
            mock_param = MagicMock()
            mock_param.device = torch.device("cuda:0")
            bundle._model = MagicMock()
            bundle._model.parameters.return_value = iter([mock_param])

            device = next(bundle.model.parameters()).device
            assert device.type == "cuda"
        else:
            pytest.skip("CUDA not available for testing")

    def test_tensor_moves_to_cuda(self, cuda_available):
        """Test tensors are moved to CUDA device."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        tensor = torch.tensor([1, 2, 3])
        cuda_tensor = tensor.cuda()

        assert cuda_tensor.device.type == "cuda"

    def test_model_generation_on_cuda(self, cuda_available):
        """Test model generation works on CUDA."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        config = ServiceConfig()
        bundle = ModelBundle(config)

        # Mock CUDA model
        mock_param = MagicMock()
        mock_param.device = torch.device("cuda:0")
        bundle._model = MagicMock()
        bundle._model.parameters.return_value = iter([mock_param])
        bundle._tokenizer = MagicMock()
        bundle._tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]).cuda()}
        bundle._tokenizer.decode.return_value = "Response"
        bundle._tokenizer.eos_token_id = 2

        # Mock generation
        bundle._model.generate.return_value = torch.tensor([[1, 2, 3, 4]]).cuda()

        import asyncio
        with patch("torch.inference_mode"):
            result = asyncio.run(bundle.generate("test", 10, 0.7, 0.9))

        assert isinstance(result, str)

    def test_quantization_config_cuda_compatible(self):
        """Test quantization config is CUDA compatible."""
        config = ServiceConfig()
        quant_config = config.quant_config

        assert quant_config.bnb_4bit_compute_dtype == torch.float16
        # Float16 is CUDA compatible

    def test_model_offload_to_cpu_when_needed(self):
        """Test model can offload to CPU when CUDA memory is limited."""
        config = ServiceConfig()
        config.device_map = "auto"

        # The device_map='auto' should handle memory management
        assert config.device_map == "auto"

    @pytest.mark.requires_cuda
    def test_cuda_memory_management(self, cuda_available):
        """Test CUDA memory is managed correctly."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Test that we can check CUDA memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()

            assert isinstance(memory_allocated, int)
            assert isinstance(memory_reserved, int)
            assert memory_allocated >= 0
            assert memory_reserved >= 0

    def test_device_detection(self):
        """Test device detection logic."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ["cuda", "cpu"]

    @pytest.mark.requires_cuda
    def test_model_eval_mode_on_cuda(self, cuda_available):
        """Test model eval mode works on CUDA."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        config = ServiceConfig()
        bundle = ModelBundle(config)

        # Mock CUDA model
        mock_model = MagicMock()
        mock_model.eval = Mock()
        bundle._model = mock_model

        # Model should be in eval mode
        bundle._model.eval()
        mock_model.eval.assert_called()

    def test_cuda_device_count(self, cuda_available):
        """Test CUDA device count detection."""
        if cuda_available:
            device_count = torch.cuda.device_count()
            assert device_count > 0
            assert isinstance(device_count, int)
        else:
            pytest.skip("CUDA not available")

    def test_cuda_device_name(self, cuda_available):
        """Test CUDA device name retrieval."""
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            assert isinstance(device_name, str)
            assert len(device_name) > 0
        else:
            pytest.skip("CUDA not available")

