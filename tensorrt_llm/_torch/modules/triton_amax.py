"""
amax.py - Triton-based Absolute Maximum (Amax) Implementation

High-performance global amax reduction using Triton kernels with autotuning.

Usage:
    from tensorrt_llm._torch.modules.amax import triton_amax
    
    # Basic usage - returns global amax as scalar tensor
    global_amax = triton_amax(input_tensor)
    
    # With specific configuration
    global_amax = triton_amax(input_tensor, config="B8192_W32")
"""

import torch
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


###############################################################################
# Kernel Configuration Registry
###############################################################################

KERNEL_CONFIGS = {
    "B4096_W8":   {"block_size": 4096,  "num_warps": 8,  "description": "512 threads, 8 elem/thread"},
    "B8192_W16":  {"block_size": 8192,  "num_warps": 16, "description": "512 threads, 16 elem/thread"},
    "B8192_W32":  {"block_size": 8192,  "num_warps": 32, "description": "1024 threads, 8 elem/thread"},
    "B16384_W32": {"block_size": 16384, "num_warps": 32, "description": "1024 threads, 16 elem/thread"},
    "B16384_W16": {"block_size": 16384, "num_warps": 16, "description": "512 threads, 32 elem/thread"},
}


###############################################################################
# Triton Kernels
###############################################################################

if TRITON_AVAILABLE:

    # Autotune configurations
    AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_SIZE": cfg["block_size"]}, num_warps=cfg["num_warps"])
        for cfg in KERNEL_CONFIGS.values()
    ]

    @triton.autotune(configs=AUTOTUNE_CONFIGS, key=["n_elements"])
    @triton.jit
    def _amax_kernel_autotune(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Autotuned amax kernel - automatically selects best configuration."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        local_max = tl.max(tl.abs(x))
        tl.store(output_ptr + pid, local_max)

    @triton.jit
    def _amax_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Generic amax kernel with configurable BLOCK_SIZE.
        
        Args:
            input_ptr: Pointer to input tensor (BF16/FP16/FP32)
            output_ptr: Pointer to output tensor (FP32, size = num_blocks)
            n_elements: Total number of elements in input
            BLOCK_SIZE: Number of elements processed per block
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        local_max = tl.max(tl.abs(x))
        tl.store(output_ptr + pid, local_max)


###############################################################################
# Public API
###############################################################################

def triton_amax(
    input_tensor: torch.Tensor,
    config: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute global absolute maximum using Triton kernel.
    
    This function performs a two-stage reduction:
    1. Triton kernel: parallel block-level reduction
    2. PyTorch: final reduction over partial results
    
    Args:
        input_tensor: Input tensor (BF16/FP16/FP32) on CUDA device
        config: Optional kernel configuration name. If None, uses autotuned kernel.
                Available configs: "B4096_W8", "B8192_W16", "B8192_W32", 
                                   "B16384_W32", "B16384_W16"
    
    Returns:
        Scalar tensor containing the global absolute maximum (FP32)
    
    Raises:
        RuntimeError: If Triton is not available
        ValueError: If invalid config name is provided
    
    Example:
        >>> x = torch.randn(14400, 6144, device="cuda", dtype=torch.bfloat16)
        >>> amax = triton_amax(x)
        >>> print(f"Global amax: {amax.item():.6f}")
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for triton_amax. Install with: pip install triton")
    
    if not input_tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    
    n_elements = input_tensor.numel()
    
    if config is None:
        # Use autotuned kernel
        # Use smallest BLOCK_SIZE to ensure all elements are covered regardless of autotune choice
        min_block_size = min(cfg["block_size"] for cfg in KERNEL_CONFIGS.values())
        grid_size = triton.cdiv(n_elements, min_block_size)
        output = torch.empty(grid_size, device=input_tensor.device, dtype=torch.float32)
        
        _amax_kernel_autotune[(grid_size,)](input_tensor, output, n_elements)
    else:
        # Use specified configuration
        if config not in KERNEL_CONFIGS:
            raise ValueError(
                f"Invalid config '{config}'. "
                f"Available configs: {list(KERNEL_CONFIGS.keys())}"
            )
        
        cfg = KERNEL_CONFIGS[config]
        block_size = cfg["block_size"]
        num_warps = cfg["num_warps"]
        
        grid_size = triton.cdiv(n_elements, block_size)
        output = torch.empty(grid_size, device=input_tensor.device, dtype=torch.float32)
        
        _amax_kernel[(grid_size,)](
            input_tensor, output, n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    
    # Final reduction with PyTorch
    return output.max()


def triton_amax_partial(
    input_tensor: torch.Tensor,
    config: str = "B8192_W32",
) -> torch.Tensor:
    """
    Compute partial absolute maximum (block-level reduction only).
    
    Returns partial results without final reduction. Useful for custom
    reduction strategies or fused operations.
    
    Args:
        input_tensor: Input tensor (BF16/FP16/FP32) on CUDA device
        config: Kernel configuration name (default: "B8192_W32")
    
    Returns:
        Tensor of shape (num_blocks,) containing per-block amax values (FP32)
    
    Example:
        >>> x = torch.randn(14400, 6144, device="cuda", dtype=torch.bfloat16)
        >>> partial_amax = triton_amax_partial(x)
        >>> global_amax = partial_amax.max()
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required. Install with: pip install triton")
    
    if config not in KERNEL_CONFIGS:
        raise ValueError(
            f"Invalid config '{config}'. "
            f"Available configs: {list(KERNEL_CONFIGS.keys())}"
        )
    
    n_elements = input_tensor.numel()
    cfg = KERNEL_CONFIGS[config]
    block_size = cfg["block_size"]
    num_warps = cfg["num_warps"]
    
    grid_size = triton.cdiv(n_elements, block_size)
    output = torch.empty(grid_size, device=input_tensor.device, dtype=torch.float32)
    
    _amax_kernel[(grid_size,)](
        input_tensor, output, n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    
    return output


###############################################################################
# Utility Functions
###############################################################################

def list_configs() -> dict:
    """
    List all available kernel configurations.
    
    Returns:
        Dictionary of config name -> description
    """
    return {name: cfg["description"] for name, cfg in KERNEL_CONFIGS.items()}


def benchmark_config(
    input_tensor: torch.Tensor,
    config: str,
    warmup: int = 5,
    repeat: int = 20,
) -> Tuple[torch.Tensor, float]:
    """
    Benchmark a specific kernel configuration.
    
    Args:
        input_tensor: Input tensor for benchmarking
        config: Configuration name to benchmark
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
    
    Returns:
        Tuple of (result_tensor, avg_time_microseconds)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for benchmarking")
    
    if config not in KERNEL_CONFIGS:
        raise ValueError(f"Invalid config '{config}'")
    
    cfg = KERNEL_CONFIGS[config]
    block_size = cfg["block_size"]
    num_warps = cfg["num_warps"]
    
    n_elements = input_tensor.numel()
    grid_size = triton.cdiv(n_elements, block_size)
    output = torch.empty(grid_size, device=input_tensor.device, dtype=torch.float32)
    
    # Warmup
    for _ in range(warmup):
        _amax_kernel[(grid_size,)](
            input_tensor, output, n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _amax_kernel[(grid_size,)](
            input_tensor, output, n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # Convert to microseconds
    
    avg_time_us = sum(times) / len(times)
    return output.max(), avg_time_us


###############################################################################
# Test Entry Point
###############################################################################

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Triton Amax Kernel Test")
    parser.add_argument("--shape", type=str, default="14400,6144", help="Input shape (comma-separated)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=20, help="Benchmark iterations")
    args = parser.parse_args()
    
    # Parse shape
    shape = tuple(map(int, args.shape.split(",")))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    print("=" * 70)
    print("Triton Amax Kernel Test")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Shape: {shape}")
    print(f"Dtype: {dtype}")
    print(f"Elements: {torch.tensor(shape).prod().item():,}")
    
    # Create input
    x = torch.randn(shape, device="cuda", dtype=dtype)
    n_elements = x.numel()
    read_bytes = n_elements * x.element_size()
    print(f"Data size: {read_bytes / 1024 / 1024:.2f} MB")
    
    # Correctness verification
    print("\n--- Correctness Verification ---")
    ref_amax = x.abs().max().float()
    print(f"PyTorch reference: {ref_amax.item():.6f}")
    
    triton_result = triton_amax(x)
    match = torch.allclose(triton_result, ref_amax, rtol=1e-3, atol=1e-5)
    status = "PASS" if match else "FAIL"
    print(f"Triton (autotune): {triton_result.item():.6f} [{status}]")
    
    for config in KERNEL_CONFIGS:
        result = triton_amax(x, config=config)
        match = torch.allclose(result, ref_amax, rtol=1e-3, atol=1e-5)
        status = "PASS" if match else "FAIL"
        print(f"Triton ({config}): {result.item():.6f} [{status}]")
    
    # Performance benchmark
    print(f"\n--- Performance Benchmark (warmup={args.warmup}, repeat={args.repeat}) ---")
    print(f"{'Config':<16} {'Time(us)':>12} {'BW(TB/s)':>12} {'MBU%':>8}")
    print("-" * 50)
    
    results = []
    for config in KERNEL_CONFIGS:
        _, time_us = benchmark_config(x, config, warmup=args.warmup, repeat=args.repeat)
        # Total bandwidth = read (input) + write (partial output)
        cfg = KERNEL_CONFIGS[config]
        grid_size = triton.cdiv(n_elements, cfg["block_size"])
        write_bytes = grid_size * 4  # float32 output
        total_bytes = read_bytes + write_bytes
        bandwidth = total_bytes / (time_us * 1e-6) / 1e12
        mbu = bandwidth / 7.7 * 100  # B200 peak = 7.7 TB/s
        results.append((config, time_us, bandwidth, mbu))
        print(f"{config:<16} {time_us:>12.2f} {bandwidth:>12.2f} {mbu:>7.1f}%")
    
    # Summary
    best = min(results, key=lambda r: r[1])
    print("-" * 50)
    print(f"Best config: {best[0]} ({best[1]:.2f} us, {best[2]:.2f} TB/s)")
    print("=" * 70)
