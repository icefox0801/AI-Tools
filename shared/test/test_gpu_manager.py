"""Unit tests for GPU memory manager."""

import unittest
from unittest.mock import MagicMock, patch

from shared.core.gpu_manager import (
    GPUMemoryManager,
    clear_gpu_cache,
    get_free_memory_gb,
    get_gpu_manager,
    get_gpu_memory_info,
)


class TestGPUManagerUtilities(unittest.TestCase):
    """Test GPU utility functions."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=2 * 1024**3)
    @patch("torch.cuda.get_device_properties")
    def test_get_gpu_memory_info_cuda_available(self, mock_props, mock_alloc, mock_avail):
        """Test GPU memory info when CUDA is available."""
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)

        used, total = get_gpu_memory_info()

        self.assertAlmostEqual(used, 2.0, places=1)
        self.assertAlmostEqual(total, 16.0, places=1)

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_gpu_memory_info_no_cuda(self, mock_avail):
        """Test GPU memory info when CUDA is not available."""
        used, total = get_gpu_memory_info()

        self.assertEqual(used, 0.0)
        self.assertEqual(total, 0.0)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_reserved", return_value=10 * 1024**3)
    @patch("torch.cuda.memory_allocated", return_value=6 * 1024**3)
    def test_get_free_memory_gb(self, mock_alloc, mock_reserved, mock_avail):
        """Test free memory calculation."""
        free = get_free_memory_gb()

        self.assertAlmostEqual(free, 4.0, places=1)  # 10 - 6 = 4GB free

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_clear_gpu_cache(self, mock_empty, mock_avail):
        """Test GPU cache clearing."""
        clear_gpu_cache()

        mock_empty.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_clear_gpu_cache_no_cuda(self, mock_avail):
        """Test cache clearing when CUDA not available."""
        # Should not raise exception
        clear_gpu_cache()


class TestGPUMemoryManager(unittest.TestCase):
    """Test GPUMemoryManager class."""

    def setUp(self):
        """Set up test manager."""
        self.manager = GPUMemoryManager()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=4 * 1024**3)
    @patch("torch.cuda.get_device_properties")
    def test_get_gpu_memory(self, mock_props, mock_alloc, mock_avail):
        """Test internal GPU memory getter."""
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)

        used, total = self.manager._get_gpu_memory()

        self.assertAlmostEqual(used, 4.0, places=1)
        self.assertAlmostEqual(total, 16.0, places=1)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=4 * 1024**3)
    @patch("torch.cuda.get_device_properties")
    def test_get_available_memory(self, mock_props, mock_alloc, mock_avail):
        """Test available memory calculation."""
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)

        available = self.manager._get_available_memory()

        # 16 - 4 - 1 (reserve) = 11GB
        self.assertAlmostEqual(available, 11.0, places=1)

    def test_register_model(self):
        """Test model registration."""
        callback = MagicMock()
        self.manager.register_model("test-service", "test-model", 5.0, callback)

        key = "test-service:test-model"
        self.assertIn(key, self.manager._loaded_models)
        self.assertEqual(self.manager._loaded_models[key].service_name, "test-service")
        self.assertEqual(self.manager._loaded_models[key].model_name, "test-model")
        self.assertEqual(self.manager._loaded_models[key].memory_gb, 5.0)
        self.assertEqual(self.manager._loaded_models[key].unload_callback, callback)

    def test_unregister_model(self):
        """Test model unregistration."""
        self.manager.register_model("test-service", "test-model", 5.0)
        self.assertEqual(len(self.manager._loaded_models), 1)

        self.manager.unregister_model("test-service", "test-model")

        self.assertEqual(len(self.manager._loaded_models), 0)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=4 * 1024**3)
    @patch("torch.cuda.get_device_properties")
    def test_request_memory_sufficient(self, mock_props, mock_alloc, mock_avail):
        """Test memory request when sufficient memory available."""
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)

        # Request 5GB (11GB available - 1GB reserve)
        result = self.manager.request_memory("test-service", "test-model", 5.0)

        self.assertTrue(result)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=10 * 1024**3)
    @patch("torch.cuda.get_device_properties")
    def test_request_memory_insufficient_no_models(self, mock_props, mock_alloc, mock_avail):
        """Test memory request fails when insufficient and no models to unload."""
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)

        # Request 10GB (5GB available - 1GB reserve)
        result = self.manager.request_memory("test-service", "test-model", 10.0)

        self.assertFalse(result)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.get_device_properties")
    @patch("shared.core.gpu_manager.requests")
    def test_request_memory_unload_other_service(
        self, mock_requests, mock_props, mock_alloc, mock_avail
    ):
        """Test memory request unloads other service models."""
        # First call: 10GB used, second call: 4GB used (after unload)
        mock_alloc.side_effect = [10 * 1024**3, 4 * 1024**3]
        mock_props.return_value = MagicMock(total_memory=16 * 1024**3)

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response

        # Register model from another service
        self.manager.register_model("other-service", "other-model", 6.0)

        # Request memory for current service (needs to unload other service)
        result = self.manager.request_memory("test-service", "test-model", 10.0)

        self.assertTrue(result)
        mock_requests.post.assert_called_once_with(
            "http://other-service:8000/unload",
            timeout=5,
        )

    def test_get_status(self):
        """Test status retrieval."""
        self.manager.register_model("service-1", "model-1", 5.0)
        self.manager.register_model("service-2", "model-2", 3.0)

        status = self.manager.get_status()

        self.assertIn("gpu_total_gb", status)
        self.assertIn("gpu_used_gb", status)
        self.assertIn("gpu_available_gb", status)
        self.assertIn("loaded_models", status)
        self.assertEqual(len(status["loaded_models"]), 2)

    def test_global_instance(self):
        """Test global GPU manager instance."""
        mgr1 = get_gpu_manager()
        mgr2 = get_gpu_manager()

        # Should be same instance
        self.assertIs(mgr1, mgr2)


if __name__ == "__main__":
    unittest.main()
