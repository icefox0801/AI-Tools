"""
Shared GPU Memory Manager

Coordinates GPU memory usage across all services to prevent OOM errors.
Services register their models and request memory before loading.
Automatically unloads other models when needed.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    service_name: str  # e.g., "whisper-asr", "parakeet-asr"
    model_name: str  # e.g., "streaming", "offline", "turbo", "large-v3"
    memory_gb: float  # Estimated GPU memory in GB
    unload_callback: Callable | None = None  # Function to unload this model


class GPUMemoryManager:
    """Centralized GPU memory manager."""

    def __init__(self):
        self._loaded_models: dict[str, ModelInfo] = {}  # key: f"{service}:{model}"
        self._gpu_total_gb: float | None = None
        self._reserve_gb = 1.0  # Reserve 1GB for operations

    def _get_gpu_memory(self) -> tuple[float, float]:
        """Get GPU memory usage in GB.

        Returns:
            (used_gb, total_gb)
        """
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return allocated, total
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
            return 0.0, 0.0

    def _get_available_memory(self) -> float:
        """Get available GPU memory in GB."""
        used, total = self._get_gpu_memory()
        if self._gpu_total_gb is None:
            self._gpu_total_gb = total
        return total - used - self._reserve_gb

    def register_model(
        self,
        service_name: str,
        model_name: str,
        memory_gb: float,
        unload_callback: Callable | None = None,
    ):
        """Register a loaded model.

        Args:
            service_name: Service name (e.g., "parakeet-asr")
            model_name: Model identifier (e.g., "streaming", "offline")
            memory_gb: GPU memory used by this model
            unload_callback: Function to call to unload this model
        """
        key = f"{service_name}:{model_name}"
        self._loaded_models[key] = ModelInfo(
            service_name=service_name,
            model_name=model_name,
            memory_gb=memory_gb,
            unload_callback=unload_callback,
        )
        logger.info(
            f"Registered model {key} ({memory_gb:.2f}GB). "
            f"Total loaded: {len(self._loaded_models)} models"
        )

    def unregister_model(self, service_name: str, model_name: str):
        """Unregister a model (called after unloading)."""
        key = f"{service_name}:{model_name}"
        if key in self._loaded_models:
            del self._loaded_models[key]
            logger.info(f"Unregistered model {key}")

    def request_memory(self, service_name: str, model_name: str, required_gb: float) -> bool:
        """Request GPU memory for loading a model.

        Automatically unloads other models if needed to free space.

        Args:
            service_name: Service requesting memory
            model_name: Model to be loaded
            required_gb: GPU memory required in GB

        Returns:
            True if memory is available or freed successfully
        """
        available = self._get_available_memory()
        logger.info(
            f"{service_name} requesting {required_gb:.2f}GB for {model_name}. "
            f"Available: {available:.2f}GB"
        )

        if available >= required_gb:
            return True

        # Need to free memory - unload models from OTHER services first
        needed = required_gb - available
        logger.warning(
            f"Insufficient GPU memory. Need to free {needed:.2f}GB. "
            f"Currently loaded: {list(self._loaded_models.keys())}"
        )

        # Sort models by memory size (largest first) and unload from other services
        other_service_models = [
            (key, info)
            for key, info in self._loaded_models.items()
            if info.service_name != service_name
        ]
        other_service_models.sort(key=lambda x: x[1].memory_gb, reverse=True)

        freed = 0.0
        for key, info in other_service_models:
            if freed >= needed:
                break

            logger.info(f"Unloading {key} from {info.service_name} to free {info.memory_gb:.2f}GB")

            # Try to unload via HTTP API first
            if self._unload_via_http(info.service_name):
                freed += info.memory_gb
                self.unregister_model(info.service_name, info.model_name)
            # Fallback to local callback
            elif info.unload_callback:
                try:
                    info.unload_callback()
                    freed += info.memory_gb
                    self.unregister_model(info.service_name, info.model_name)
                except Exception as e:
                    logger.error(f"Failed to unload {key} via callback: {e}")

        # Check if we freed enough
        available = self._get_available_memory()
        if available >= required_gb:
            logger.info(f"Successfully freed {freed:.2f}GB. Available: {available:.2f}GB")
            return True

        logger.error(
            f"Failed to free enough memory. Required: {required_gb:.2f}GB, "
            f"Available: {available:.2f}GB"
        )
        return False

    def _unload_via_http(self, service_name: str) -> bool:
        """Try to unload models via HTTP API.

        Args:
            service_name: Service name (e.g., "parakeet-asr", "whisper-asr")

        Returns:
            True if successful
        """
        try:
            # Service names match container names
            url = f"http://{service_name}:8000/unload"
            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully unloaded models from {service_name} via HTTP")
                return True
            logger.warning(f"Failed to unload {service_name} via HTTP: {response.status_code}")
            return False
        except Exception as e:
            logger.warning(f"HTTP unload failed for {service_name}: {e}")
            return False

    def get_status(self) -> dict:
        """Get current GPU memory status.

        Returns:
            Dict with memory info and loaded models
        """
        used, total = self._get_gpu_memory()
        return {
            "gpu_total_gb": total,
            "gpu_used_gb": used,
            "gpu_available_gb": total - used,
            "reserve_gb": self._reserve_gb,
            "loaded_models": {
                key: {
                    "service": info.service_name,
                    "model": info.model_name,
                    "memory_gb": info.memory_gb,
                }
                for key, info in self._loaded_models.items()
            },
        }


# Utility functions for common GPU operations


def get_gpu_memory_info() -> tuple[float, float]:
    """Get current GPU memory usage.

    Returns:
        (used_gb, total_gb) tuple
    """
    try:
        import torch

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return used, total
        return 0.0, 0.0
    except Exception:
        return 0.0, 0.0


def get_free_memory_gb() -> float:
    """Get available GPU memory in GB."""
    try:
        import torch

        if torch.cuda.is_available():
            free = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            return free / 1024**3
        return 0.0
    except Exception:
        return 0.0


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Global instance
_gpu_manager = GPUMemoryManager()


def get_gpu_manager() -> GPUMemoryManager:
    """Get the global GPU memory manager instance."""
    return _gpu_manager
