"""
GPU Memory Manager

Coordinates GPU memory between ASR and LLM services.
Provides functions to unload models and check memory before loading new ones.

Usage:
    from shared.gpu_memory import GPUMemoryManager
    
    manager = GPUMemoryManager(
        parakeet_url="http://parakeet-asr:8000",
        whisper_url="http://whisper-asr:8000"
    )
    
    # Before summarization with LLM
    await manager.prepare_for_llm()
    
    # Check memory status
    status = await manager.get_status()
"""

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default service URLs
DEFAULT_PARAKEET_URL = "http://parakeet-asr:8000"
DEFAULT_WHISPER_URL = "http://whisper-asr:8000"

# Memory thresholds (fraction of total GPU memory)
GPU_MEMORY_HIGH_THRESHOLD = 0.80  # Unload ASR if >80% used before loading LLM


@dataclass
class ServiceStatus:
    """Status of a GPU service."""
    name: str
    available: bool
    model_loaded: bool
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_usage: float = 0.0
    active_sessions: int = 0
    error: Optional[str] = None


@dataclass 
class GPUStatus:
    """Overall GPU memory status."""
    total_gb: float
    used_gb: float
    usage: float
    services: list[ServiceStatus]


class GPUMemoryManager:
    """
    Manages GPU memory coordination between ASR and LLM services.
    
    This class helps ensure there's enough GPU memory available
    before loading LLM models for summarization by unloading
    idle ASR models.
    """
    
    def __init__(
        self,
        parakeet_url: str = DEFAULT_PARAKEET_URL,
        whisper_url: str = DEFAULT_WHISPER_URL,
        timeout: float = 30.0
    ):
        self.parakeet_url = parakeet_url
        self.whisper_url = whisper_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def _get_service_status(self, url: str, name: str) -> ServiceStatus:
        """Get status of a single service."""
        try:
            resp = await self.client.get(f"{url}/health")
            if resp.status_code == 200:
                data = resp.json()
                return ServiceStatus(
                    name=name,
                    available=True,
                    model_loaded=data.get("model_loaded", False),
                    memory_used_gb=data.get("memory_used_gb", data.get("memory_gb", 0.0)),
                    memory_total_gb=data.get("memory_total_gb", 0.0),
                    memory_usage=data.get("memory_usage", 0.0),
                    active_sessions=data.get("active_sessions", 0),
                )
            else:
                return ServiceStatus(
                    name=name,
                    available=False,
                    model_loaded=False,
                    error=f"HTTP {resp.status_code}"
                )
        except Exception as e:
            return ServiceStatus(
                name=name,
                available=False,
                model_loaded=False,
                error=str(e)
            )
    
    async def _unload_service(self, url: str, name: str) -> tuple[bool, str]:
        """Unload model from a service."""
        try:
            resp = await self.client.post(f"{url}/unload")
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "unknown")
                message = data.get("message", "")
                logger.info(f"Unloaded {name}: {status} - {message}")
                return True, message
            else:
                logger.warning(f"Failed to unload {name}: HTTP {resp.status_code}")
                return False, f"HTTP {resp.status_code}"
        except Exception as e:
            logger.warning(f"Could not unload {name}: {e}")
            return False, str(e)
    
    async def get_status(self) -> GPUStatus:
        """Get status of all GPU services."""
        parakeet_status = await self._get_service_status(self.parakeet_url, "parakeet")
        whisper_status = await self._get_service_status(self.whisper_url, "whisper")
        
        services = [parakeet_status, whisper_status]
        
        # Calculate total GPU usage from services
        total_gb = max(s.memory_total_gb for s in services if s.available) if any(s.available for s in services) else 0.0
        used_gb = sum(s.memory_used_gb for s in services if s.available and s.model_loaded)
        usage = used_gb / total_gb if total_gb > 0 else 0.0
        
        return GPUStatus(
            total_gb=total_gb,
            used_gb=used_gb,
            usage=usage,
            services=services
        )
    
    async def unload_asr_models(self) -> dict:
        """
        Unload all ASR models to free GPU memory.
        
        Returns:
            Dict with results for each service
        """
        results = {}
        
        # Get current status
        status = await self.get_status()
        
        for service in status.services:
            if service.available and service.model_loaded and service.active_sessions == 0:
                success, message = await self._unload_service(
                    self.parakeet_url if service.name == "parakeet" else self.whisper_url,
                    service.name
                )
                results[service.name] = {
                    "unloaded": success,
                    "message": message,
                    "memory_freed_gb": service.memory_used_gb if success else 0.0
                }
            elif service.available and service.model_loaded and service.active_sessions > 0:
                results[service.name] = {
                    "unloaded": False,
                    "message": f"Model in use ({service.active_sessions} active sessions)",
                    "memory_freed_gb": 0.0
                }
            elif service.available and not service.model_loaded:
                results[service.name] = {
                    "unloaded": False,
                    "message": "Model not loaded",
                    "memory_freed_gb": 0.0
                }
            else:
                results[service.name] = {
                    "unloaded": False,
                    "message": service.error or "Service unavailable",
                    "memory_freed_gb": 0.0
                }
        
        return results
    
    async def prepare_for_llm(self, required_memory_gb: float = 6.0) -> dict:
        """
        Prepare GPU memory for loading LLM by unloading ASR models if needed.
        
        Args:
            required_memory_gb: Approximate memory needed for LLM (default 6GB for 7B model)
            
        Returns:
            Dict with preparation status and actions taken
        """
        status = await self.get_status()
        
        result = {
            "initial_usage": status.usage,
            "initial_used_gb": status.used_gb,
            "total_gb": status.total_gb,
            "required_gb": required_memory_gb,
            "actions": [],
            "ready": False
        }
        
        # Check if we have enough free memory
        free_gb = status.total_gb - status.used_gb
        if free_gb >= required_memory_gb:
            result["ready"] = True
            result["message"] = f"Sufficient GPU memory available ({free_gb:.1f}GB free)"
            return result
        
        # Need to free memory - unload ASR models
        logger.info(f"Need {required_memory_gb:.1f}GB, only {free_gb:.1f}GB free. Unloading ASR models...")
        
        unload_results = await self.unload_asr_models()
        result["actions"] = unload_results
        
        total_freed = sum(r.get("memory_freed_gb", 0) for r in unload_results.values())
        
        # Re-check status
        new_status = await self.get_status()
        new_free_gb = new_status.total_gb - new_status.used_gb
        
        result["final_usage"] = new_status.usage
        result["final_used_gb"] = new_status.used_gb
        result["memory_freed_gb"] = total_freed
        result["ready"] = new_free_gb >= required_memory_gb
        
        if result["ready"]:
            result["message"] = f"GPU memory prepared. Freed {total_freed:.1f}GB, now {new_free_gb:.1f}GB available."
        else:
            result["message"] = f"Could not free enough memory. Need {required_memory_gb:.1f}GB, only {new_free_gb:.1f}GB available."
        
        logger.info(result["message"])
        return result


# Singleton instance for convenience
_manager: Optional[GPUMemoryManager] = None


def get_manager(
    parakeet_url: str = DEFAULT_PARAKEET_URL,
    whisper_url: str = DEFAULT_WHISPER_URL
) -> GPUMemoryManager:
    """Get or create the GPU memory manager singleton."""
    global _manager
    if _manager is None:
        _manager = GPUMemoryManager(
            parakeet_url=parakeet_url,
            whisper_url=whisper_url
        )
    return _manager
