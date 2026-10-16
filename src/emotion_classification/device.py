import platform

import torch


def select_device(allow_mps=True):
    """Return the best available device: CUDA, then MPS (Apple silicon, if allowed), then CPU."""
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and \
                    platform.processor() in {"arm", "arm64"}
    device = torch.device("cuda" if cuda_available else "mps" if (mps_available and allow_mps) else "cpu")
    print(f"Device: {device}")
    return device
