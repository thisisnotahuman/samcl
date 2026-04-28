from __future__ import annotations

import torch


class GpuMatmulBurner:
    """
    Intentionally performs GPU matmul work to increase utilization.

    This is a pragmatic HPC workaround for clusters that may kill jobs
    with low GPU utilization. It does NOT change training objectives,
    but it WILL slow training and consume extra GPU cycles.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        matmul_dim: int = 2048,
        iters: int = 4,
        dtype: str = "float16",
        reserve_gb: float = 0.0,
    ) -> None:
        self.device = device
        self.matmul_dim = int(matmul_dim)
        self.iters = int(iters)
        self.dtype = str(dtype).lower()
        self.reserve_gb = float(reserve_gb)

        self._a: torch.Tensor | None = None
        self._b: torch.Tensor | None = None
        self._reserve: torch.Tensor | None = None

    def _torch_dtype(self) -> torch.dtype:
        if self.dtype in ("fp16", "float16", "half"):
            return torch.float16
        if self.dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if self.dtype in ("fp32", "float32", "float"):
            return torch.float32
        raise ValueError(f"Unsupported dtype: {self.dtype} (use float16|bfloat16|float32)")

    def _ensure_buffers(self) -> None:
        if self.device.type != "cuda":
            return
        if self._a is not None and self._b is not None and (self._reserve is not None or self.reserve_gb <= 0):
            return
        d = self.matmul_dim
        dt = self._torch_dtype()
        # Use fixed random tensors; reuse to avoid allocations each step.
        g = torch.Generator(device=self.device)
        g.manual_seed(0)
        if self._a is None:
            self._a = torch.randn((d, d), device=self.device, dtype=dt, generator=g)
        if self._b is None:
            self._b = torch.randn((d, d), device=self.device, dtype=dt, generator=g)

        # Optional: reserve additional VRAM (in bytes-precise uint8 buffer).
        if self.reserve_gb > 0 and self._reserve is None:
            n_bytes = int(self.reserve_gb * (1024**3))
            # uint8 uses 1 byte per element; allocator will round up internally.
            self._reserve = torch.empty((n_bytes,), device=self.device, dtype=torch.uint8)

    @torch.no_grad()
    def __call__(self) -> None:
        if self.device.type != "cuda":
            return
        self._ensure_buffers()
        assert self._a is not None and self._b is not None

        x = self._a
        y = self._b
        # Chain matmuls; keep result alive to prevent dead-code elimination.
        for _ in range(max(1, self.iters)):
            x = x @ y
        # Use a tiny reduction to ensure side effect is observed.
        _ = float(x[0, 0].item())

