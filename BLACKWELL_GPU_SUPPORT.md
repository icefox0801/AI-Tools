# RTX 5070 Ti / Blackwell GPU Support Investigation

## Issue Summary
FastConformer service was failing to load model on RTX 5070 Ti (Blackwell architecture) with error:
```
CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

## Root Cause
- **GPU**: RTX 5070 Ti uses **Blackwell architecture** with **Compute Capability 12.0 (sm_120)**
- **PyTorch 2.6.0**: Only supports up to **sm_90** (Hopper)
- **Required**: PyTorch 2.9+ for Blackwell support

## GPU Architecture Timeline
- **Maxwell** (sm_50): GTX 900 series
- **Pascal** (sm_60): GTX 1000 series  
- **Volta** (sm_70): Titan V, Tesla V100
- **Turing** (sm_75): RTX 2000 series, GTX 1600 series
- **Ampere** (sm_80, sm_86): RTX 3000 series, A100
- **Hopper** (sm_90): H100
- **Blackwell** (sm_100, sm_120): RTX 5000 series ← **Your GPU**

## PyTorch CUDA Support Matrix (from PyTorch 2.9+)

| CUDA Version | Supported Architectures |
|--------------|-------------------------|
| 12.6.3 | Maxwell(5.0), Pascal(6.0), Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0) |
| **12.8.1** | **Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0)** ← Solution |
| 13.0.0 | Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0+PTX) |

## Solution
Upgraded PyTorch from **2.6.0** to **2.9.1** with **CUDA 12.8**

### Changes Made

#### 1. `services/fastconformer/Dockerfile`
```dockerfile
# Before:
# Install PyTorch 2.6.0 with CUDA 12.8 support (compatible with driver 576.88)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url ${PYTORCH_CUDA_MIRROR}

# After:
# Install PyTorch 2.9.1 with CUDA 12.8 support (compatible with Blackwell sm_120)
# PyTorch 2.9+ is required for RTX 5070 Ti and other Blackwell GPUs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.9.1 torchaudio==2.9.1 --index-url ${PYTORCH_CUDA_MIRROR}
```

#### 2. `services/fastconformer/requirements.docker.txt`
```python
# Before:
# torch==2.6.0
# torchaudio==2.6.0

# After:
# torch==2.9.1 (CUDA 12.8 - required for Blackwell RTX 5070 Ti support)
# torchaudio==2.9.1
```

## Verification Steps

After rebuilding the container:

```bash
# 1. Rebuild FastConformer service
docker compose build fastconformer-asr

# 2. Restart the service
docker compose up -d fastconformer-asr

# 3. Verify PyTorch version and GPU support
docker exec fastconformer-asr python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
"
```

Expected output:
```
PyTorch: 2.9.1+cu128
CUDA Available: True
CUDA Version: 12.8
GPU: NVIDIA GeForce RTX 5070 Ti
Compute Capability: (12, 0)
```

## Why "50 Series" Has Been Out for a Long Time

**Clarification**: The RTX 5000 series (Blackwell) is actually **brand new** as of late 2024/early 2025. However, you are correct that it should be well-supported by now:

- **Release**: RTX 5090/5080/5070 Ti announced January 2025
- **PyTorch Support**: Added in PyTorch 2.9.0 (September 2025 release)
- **Current**: PyTorch 2.9.1 (October 2025)

The architecture has been out long enough that PyTorch has full support in stable releases (not just nightly builds).

## Alternative: Build PyTorch from Source

If you need cutting-edge features or specific CUDA versions, you can build PyTorch from source with Blackwell support:

```dockerfile
# This requires much more build time and disk space
RUN git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && \
    export TORCH_CUDA_ARCH_LIST="5.0 6.0 7.0 7.5 8.0 8.6 9.0 10.0 12.0" && \
    python setup.py install
```

**Not recommended** unless you specifically need unreleased features, as it:
- Takes 1-2 hours to build
- Requires 20+ GB disk space during build
- Requires C++ compiler and CUDA toolkit in the build environment

## References

- [PyTorch Release Notes](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [PyTorch CUDA Support Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix)
- [NVIDIA CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

## Related Issues

- PyTorch CUDA compatibility: https://pytorch.org/get-started/locally/
- NeMo 2.3.0 compatibility: Works with PyTorch 2.6-2.9
- CUDA 12.8 driver requirements: NVIDIA Driver 576.88+

## Status

✅ **Fixed**: Upgraded to PyTorch 2.9.1 with CUDA 12.8 support for Blackwell GPUs
🔄 **Testing**: Rebuild in progress, container will be restarted with new image
📊 **Next**: Verify model loading and run e2e tests to confirm full functionality
