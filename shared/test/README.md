# Shared Module Tests

Unit tests for shared modules used across services.

## Running Tests

Tests require PyTorch and other dependencies. Run inside Docker containers:

```bash
# Run all tests
docker exec whisper-asr python3 -m unittest discover -s /app/shared/test -v

# Run specific test module
docker exec whisper-asr python3 -m unittest shared.test.test_gpu_manager -v

# Or from workspace root
docker exec whisper-asr sh -c 'cd /app && python3 -m unittest shared.test.test_gpu_manager -v'
```

## Test Modules

- **test_gpu_manager.py**: Tests for GPU memory manager
  - Model registration/unregistration
  - Memory request logic
  - Automatic model unloading
  - HTTP-based service coordination
  - Utility functions (get_gpu_memory_info, clear_gpu_cache, etc.)

## Notes

- Tests use mocking to avoid requiring actual GPU hardware
- Some tests patch torch.cuda functions to simulate different memory scenarios
- Tests must run in environment with PyTorch installed (not host machine)
- Follow Python naming convention: test modules start with `test_`
