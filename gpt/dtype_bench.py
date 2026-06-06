#!/usr/bin/env python3
"""
Benchmark and compare different floating-point dtypes on GPU via CuPy:
  fp64, fp32, tf32 (via Tensor Cores), bf16 (manual truncation), fp16

Shows differences in:
  - Precision / rounding behavior
  - Performance (GFLOPS) on matmul
  - Numerical error relative to fp64 reference
  - Value range limits
  - Memory bandwidth
"""

import cupy as cp
import numpy as np
import time

from cupy_backends.cuda.libs import cublas

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
M, N, K = 4096, 4096, 4096
WARMUP = 5
REPEATS = 20

# ---------------------------------------------------------------------------
# bf16 helpers — CuPy 14 does not expose cp.bfloat16 natively.
# We truncate the fp32 mantissa to 7 bits (bf16) and store in fp32.
# ---------------------------------------------------------------------------

def fp32_to_bf16(arr_f32: cp.ndarray) -> cp.ndarray:
    """Truncate fp32 mantissa to 7 bits → bf16-equivalent (stored in fp32)."""
    as_uint = arr_f32.view(cp.uint32)
    truncated = as_uint & cp.uint32(0xFFFF0000)
    return truncated.view(cp.float32)

# ---------------------------------------------------------------------------
# Dtype metadata
# ---------------------------------------------------------------------------

DTYPE_INFO = {
    "fp64": {"bits": 64, "exp_bits": 11, "frac_bits": 52, "max": np.finfo(np.float64).max},
    "fp32": {"bits": 32, "exp_bits": 8,  "frac_bits": 23, "max": float(np.finfo(np.float32).max)},
    "tf32": {"bits": 19, "exp_bits": 8,  "frac_bits": 10, "max": float(np.finfo(np.float32).max)},
    "bf16": {"bits": 16, "exp_bits": 8,  "frac_bits": 7,  "max": float(np.finfo(np.float32).max)},
    "fp16": {"bits": 16, "exp_bits": 5,  "frac_bits": 10, "max": float(np.finfo(np.float16).max)},
}

# ---------------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------------

def gflops(flops: float, seconds: float) -> float:
    return flops / seconds / 1e9


def bench_matmul(a: cp.ndarray, b: cp.ndarray, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        cp.matmul(a, b)
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        cp.matmul(a, b)
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    median_t = float(np.median(times))
    flops = 2 * M * N * K
    return median_t, gflops(flops, median_t)


def rel_error(ref: cp.ndarray, approx: cp.ndarray) -> float:
    num = cp.linalg.norm(approx.ravel().astype(cp.float64) - ref.ravel()).item()
    den = cp.linalg.norm(ref.ravel()).item()
    return num / den if den > 0 else 0.0


# ===========================================================================
# 1. Dtype properties table
# ===========================================================================
print("=" * 78)
print("1. DTYPE PROPERTIES")
print("=" * 78)
print(f"{'dtype':>8}  {'bits':>5}  {'exp':>4}  {'frac':>5}  {'max value':>18}  {'eps':>12}")
for name, info in DTYPE_INFO.items():
    eps = 2.0 ** -(info["frac_bits"] + 1)
    print(f"{name:>8}  {info['bits']:>5}  {info['exp_bits']:>4}  "
          f"{info['frac_bits']:>5}  {info['max']:>18.6e}  {eps:>12.6e}")

# ===========================================================================
# 2. Value range demo
# ===========================================================================
print()
print("=" * 78)
print("2. VALUE RANGE (overflow behavior)")
print("=" * 78)

for val in [65000.0, 66000.0]:
    arr = cp.array([val], dtype=cp.float16)
    print(f"fp16({val:>10.1f}) = {arr.item():>10.1f}")

for val in [66000.0, 1e10]:
    arr_f32 = cp.array([val], dtype=cp.float32)
    arr_bf16 = fp32_to_bf16(arr_f32)
    print(f"bf16({val:>10.1f}) -> {arr_bf16.item():>10.1f}  (no overflow — 8-bit exp)")

# ===========================================================================
# 3. Precision: smallest representable delta (ulp)
# ===========================================================================
print()
print("=" * 78)
print("3. PRECISION DEMO (ulp = unit in the last place)")
print("=" * 78)

print("At value = 1.0:")
for name, info in DTYPE_INFO.items():
    ulp = 2.0 ** -(info["frac_bits"])
    print(f"  {name:>8}:  smallest delta (ulp) = {ulp:.6e}")

print("\nAt value = 65504.0 (near fp16 max):")
print(f"  fp16   :  ulp = {2.0**-10:.6f}")
print(f"  bf16   :  ulp = {65504.0 * 2**-7:.2f}  (coarse!)")
print(f"  fp32   :  ulp = {65504.0 * 2**-23:.6e}")

print("\n1.0 + delta  (half-ulp vs full-ulp):")
tests = [
    ("fp32", 2**-24, cp.float32, None),
    ("fp32", 2**-23, cp.float32, None),
    ("fp16", 2**-11, cp.float16, None),
    ("fp16", 2**-10, cp.float16, None),
    ("bf16", 2**-8,  cp.float32, "bf16_trunc"),
    ("bf16", 2**-7,  cp.float32, "bf16_trunc"),
]
for name, delta, dtype, mode in tests:
    if mode == "bf16_trunc":
        x = fp32_to_bf16(cp.array([1.0 + delta], dtype=cp.float32))
        r = float(x.item())
    else:
        x = cp.array([1.0 + delta], dtype=dtype)
        r = float(x.item())
    diff = r - 1.0
    note = "(half ulp, should round to 1.0)" if diff == 0 else "(full ulp, representable)"
    print(f"  {name:>8}:  1.0 + {delta:.2e} = {r:.16f}  diff={diff:.2e}  {note}")

# ===========================================================================
# 4. Matmul performance
# ===========================================================================
print()
print("=" * 78)
print(f"4. MATMUL PERFORMANCE  ({M}x{K} @ {K}x{N})")
print("=" * 78)

rng = np.random.RandomState(42)
A_np = rng.randn(M, K).astype(np.float32) * 0.02
B_np = rng.randn(K, N).astype(np.float32) * 0.02

results = {}
handle = cp.cuda.Device(0).cublas_handle

# --- fp64 ---
a_f64 = cp.asarray(A_np, dtype=cp.float64)
b_f64 = cp.asarray(B_np, dtype=cp.float64)
t, g = bench_matmul(a_f64, b_f64, WARMUP, REPEATS)
results["fp64"] = {"time_s": t, "gflops": g, "result": cp.matmul(a_f64, b_f64)}

# --- fp32 (DEFAULT_MATH — no Tensor Cores, full fp32 precision) ---
a_f32 = cp.asarray(A_np, dtype=cp.float32)
b_f32 = cp.asarray(B_np, dtype=cp.float32)
cublas.setMathMode(handle, cublas.CUBLAS_DEFAULT_MATH)
t, g = bench_matmul(a_f32, b_f32, WARMUP, REPEATS)
c_fp32 = cp.matmul(a_f32, b_f32)
results["fp32"] = {"time_s": t, "gflops": g, "result": c_fp32}

# --- fp32+tf32 (TENSOR_OP_MATH — fp32 storage, Tensor Core computes in tf32 internally) ---
cublas.setMathMode(handle, cublas.CUBLAS_TENSOR_OP_MATH)
t, g = bench_matmul(a_f32, b_f32, WARMUP, REPEATS)
c_fp32_tf32 = cp.matmul(a_f32, b_f32)
cp.cuda.Device(0).synchronize()
results["fp32+tf32"] = {"time_s": t, "gflops": g, "result": c_fp32_tf32}

# --- fp16 ---
cublas.setMathMode(handle, cublas.CUBLAS_DEFAULT_MATH)
a_f16 = cp.asarray(A_np, dtype=cp.float16)
b_f16 = cp.asarray(B_np, dtype=cp.float16)
t, g = bench_matmul(a_f16, b_f16, WARMUP, REPEATS)
results["fp16"] = {"time_s": t, "gflops": g, "result": cp.matmul(a_f16, b_f16)}

# --- bf16* (truncated fp32 inputs; computed in fp32 hardware) ---
A_bf16 = fp32_to_bf16(cp.asarray(A_np, dtype=cp.float32))
B_bf16 = fp32_to_bf16(cp.asarray(B_np, dtype=cp.float32))
t, g = bench_matmul(A_bf16, B_bf16, WARMUP, REPEATS)
results["bf16*"] = {"time_s": t, "gflops": g, "result": cp.matmul(A_bf16, B_bf16)}

# Restore default
cublas.setMathMode(handle, cublas.CUBLAS_DEFAULT_MATH)

# Theoretical peaks for RTX 3060 (GA106, CC 8.6, 3584 CUDA cores @ 1777 MHz)
# fp32 (CUDA cores): 3584 × 2 FMA/clock × 1.777 GHz = 12.74 TFLOPS
# fp16 (CUDA cores): same as fp32 on GA106 (no 2× fp16 throughput on CUDA cores)
# fp16 (Tensor Core): 112 TCs × 256 FP16 FMA/TC/clock × 2 × 1.777 GHz ≈ 25.5 TFLOPS (dense)
# fp64: fp32 / 64 ≈ 0.199 TFLOPS (Ampere consumer ratio)
THEOR_PEAK = {
    "fp64":     199.1,    # GFLOPS
    "fp32":     12740.0,  # GFLOPS = 12.74 TFLOPS
    "fp32+tf32":25500.0,  # GFLOPS = 25.5 TFLOPS (Tensor Core)
    "fp16":     25500.0,  # GFLOPS = 25.5 TFLOPS (Tensor Core)
    "bf16*":    12740.0,  # GFLOPS (runs on fp32 hw)
}

# Print table
print(f"{'dtype':>8}  {'time (ms)':>10}  {'GFLOPS':>10}  {'theor peak':>12}  {'% peak':>7}  {'note'}")
print("-" * 90)
ref_g = results["fp32"]["gflops"]
for name in ["fp64", "fp32", "fp32+tf32", "fp16", "bf16*"]:
    r = results[name]
    t_ms = r["time_s"] * 1000
    peak = THEOR_PEAK.get(name, 1)
    efficiency = r["gflops"] / peak * 100
    notes = {
        "fp64":     "reference, slowest",
        "fp32":     "fp32 inputs, DEFAULT_MATH (no TC, full precision)",
        "fp32+tf32":"SAME fp32 inputs, TENSOR_OP_MATH (TC uses tf32 internally)",
        "fp16":     "half precision, Tensor Core fastest",
        "bf16*":    "bf16-precision inputs, fp32 compute",
    }
    print(f"{name:>8}  {t_ms:>10.3f}  {r['gflops']:>10.1f}  {peak:>10.1f} TF  {efficiency:>6.1f}%  {notes[name]}")

print()
print("Theoretical peaks based on RTX 3060 specs: 3584 CUDA cores @ 1777 MHz ref clock.")
print("fp16/fp32+tf32 peak = 25.5 TFLOPS (Tensor Core dense); fp32 peak = 12.74 TFLOPS (CUDA cores).")
print("70-97% of theoretical peak is normal for real-world matmul (not synthetic peak benchmark).")

# ===========================================================================
# 5. Numerical error vs fp64
# ===========================================================================
print()
print("=" * 78)
print("5. NUMERICAL ACCURACY (relative L2 error vs fp64)")
print("=" * 78)

fp64_ref = results["fp64"]["result"].ravel()
for name in ["fp32", "fp32+tf32", "fp16", "bf16*"]:
    r = results[name]
    err = rel_error(fp64_ref, r["result"].ravel())
    eps_est = {"fp32": 2**-24, "fp32+tf32": 2**-11, "fp16": 2**-11, "bf16*": 2**-8}[name]
    print(f"  {name:>8}:  rel_err = {err:.6e}  (expected order ~ {eps_est:.2e})")

# Show a few element differences directly
print("\n  Per-element comparison (first 5 elements):")
for name in ["fp32", "fp32+tf32", "fp16", "bf16*"]:
    r = results[name]
    flat_r = r["result"].ravel()
    flat_ref = fp64_ref
    diffs = [abs(float(flat_r[i]) - float(flat_ref[i])) for i in range(5)]
    print(f"  {name:>8}:  " + "  ".join(f"{d:.2e}" for d in diffs))

# ===========================================================================
# 6. Reduction error — accumulation precision
# ===========================================================================
print()
print("=" * 78)
print("6. ACCUMULATION ERROR (sum of many small values)")
print("=" * 78)

N_ACC = 100_000
acc_np = np.full(N_ACC, np.float32(1e-3))
acc_np[0] = 1.0
exact = float(np.sum(acc_np.astype(np.float64)))

for name, dtype in [("fp32", cp.float32), ("fp16", cp.float16)]:
    x = cp.asarray(acc_np, dtype=dtype)
    s = float(cp.sum(x).item())
    err = abs(s - exact) / exact
    print(f"  {name:>8}:  sum = {s:.8f}  (exact = {exact})  rel_err = {err:.2e}")

# bf16 manual
x_bf16 = fp32_to_bf16(cp.asarray(acc_np, dtype=cp.float32))
s_bf16 = float(cp.sum(x_bf16).item())
err = abs(s_bf16 - exact) / exact
print(f"  {'bf16*':>8}:  sum = {s_bf16:.8f}  (exact = {exact})  rel_err = {err:.2e}")

# ===========================================================================
# 7. Memory bandwidth — truly memory-bound ops
# ===========================================================================
print()
print("=" * 78)
print("7. MEMORY BANDWIDTH (2^28 elements, memory-bound ops)")
print("=" * 78)

# RTX 3060 12GB theoretical: 360 GB/s (192-bit GDDR6 @ 15 Gbps)
BW_THEOR = 360.0  # GB/s

N_BW = 2**28  # 268M elements

# Generate fp32, then optionally cast to fp16 (cupy.random only supports f32/f64)
x_f32 = cp.random.randn(N_BW, dtype=cp.float32)
y_f32 = cp.random.randn(N_BW, dtype=cp.float32)
z_f32 = cp.empty(N_BW, dtype=cp.float32)

print(f"{'op':>16}  {'dtype':>8}  {'data moved':>12}  {'time (ms)':>10}  {'GB/s':>10}  {'% peak':>7}")
print("-" * 78)

for dtype_name, dtype in [("fp32", cp.float32), ("fp16", cp.float16)]:
    elem_size = DTYPE_INFO[dtype_name]["bits"] // 8
    n_bytes = N_BW * elem_size

    if dtype == cp.float16:
        x = x_f32.astype(cp.float16)
        y = y_f32.astype(cp.float16)
        z = cp.empty(N_BW, dtype=cp.float16)
    else:
        x, y, z = x_f32, y_f32, z_f32

    # --- vector copy: y[:] = x  (1 read + 1 write = 2× elem) ---
    for _ in range(3):
        z[:] = x
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        z[:] = x
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    t_copy = float(np.median(times))
    bw_copy = (2 * n_bytes) / t_copy / 1e9
    print(f"{'copy (z=x)':>16}  {dtype_name:>8}  {2*n_bytes:>12,d}B  {t_copy*1000:>10.3f}  "
          f"{bw_copy:>10.1f}  {bw_copy/BW_THEOR*100:>6.1f}%")

    # --- scale: z = 2.0 * x  (1 read + 1 write) ---
    for _ in range(3):
        cp.multiply(x, 2.0, out=z)
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        cp.multiply(x, 2.0, out=z)
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    t_scale = float(np.median(times))
    bw_scale = (2 * n_bytes) / t_scale / 1e9
    print(f"{'scale (z=2*x)':>16}  {dtype_name:>8}  {2*n_bytes:>12,d}B  {t_scale*1000:>10.3f}  "
          f"{bw_scale:>10.1f}  {bw_scale/BW_THEOR*100:>6.1f}%")

    # --- add: z = x + y  (2 reads + 1 write = 3× elem) ---
    for _ in range(3):
        cp.add(x, y, out=z)
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        cp.add(x, y, out=z)
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    t_add = float(np.median(times))
    bw_add = (3 * n_bytes) / t_add / 1e9
    print(f"{'add (z=x+y)':>16}  {dtype_name:>8}  {3*n_bytes:>12,d}B  {t_add*1000:>10.3f}  "
          f"{bw_add:>10.1f}  {bw_add/BW_THEOR*100:>6.1f}%")

    # --- triad: z = x + 3.0 * y  (2 reads + 1 write = 3× elem) ---
    for _ in range(3):
        z[:] = x + 3.0 * y
    cp.cuda.Device(0).synchronize()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        z[:] = x + 3.0 * y
        cp.cuda.Device(0).synchronize()
        times.append(time.perf_counter() - t0)
    t_triad = float(np.median(times))
    bw_triad = (3 * n_bytes) / t_triad / 1e9
    print(f"{'triad (z=x+3y)':>16}  {dtype_name:>8}  {3*n_bytes:>12,d}B  {t_triad*1000:>10.3f}  "
          f"{bw_triad:>10.1f}  {bw_triad/BW_THEOR*100:>6.1f}%")

print()
print(f"RTX 3060 theoretical memory bandwidth: {BW_THEOR} GB/s (192-bit GDDR6 @ 15 Gbps).")
print(f"Simple copy/scale can reach 80-90% of peak; fused ops (add/triad) saturate the bus.")

# ===========================================================================
# 8. GPU info
# ===========================================================================
print()
print("=" * 78)
print("8. GPU INFO")
print("=" * 78)
dev = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(0)
cc_val = int(dev.compute_capability)
print(f"Device            : {props['name'].decode()}")
print(f"Compute Capability: {cc_val}")
print(f"Total memory      : {props['totalGlobalMem'] / 1e9:.1f} GB")
print(f"CuPy version      : {cp.__version__}")
print(f"CUDA version      : {cp.cuda.runtime.runtimeGetVersion()}")

print()
for feat, min_cc, note in [
    ("fp16 (native)",      60, "Pascal+"),
    ("bf16 (native)",      80, "Ampere+"),
    ("tf32 (Tensor Core)", 80, "Ampere+"),
]:
    status = "YES" if cc_val >= min_cc else "NO"
    print(f"  {feat:>20s}: {status:>4s}  (CC {cc_val} >= {min_cc}, {note})")

# ===========================================================================
# 9. Summary
# ===========================================================================
print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("""
  fp64   — 64-bit IEEE.  11-bit exp, 52-bit fraction.
            Slowest on consumer GPUs.  Use for: HPC, scientific computing.

  fp32   — 32-bit IEEE.  8-bit exp, 23-bit fraction.
            Standard ML training precision.  Default in most frameworks.

  tf32   — NVIDIA TensorFloat-32 (Ampere+).  Stored in fp32, computed with
            19-bit internal format (10-bit fraction) on Tensor Cores.
            ~8x faster matmul than fp32 on A100.  Same range as fp32.
            Use for: training — most frameworks enable it by default.

  bf16   — Brain Float 16.  8-bit exp, 7-bit fraction.
            Same range as fp32 (no overflow), less precision.
            2 bytes/element (half of fp32).
            Use for: training without loss scaling (BERT, GPT-style models).

  fp16   — IEEE half precision.  5-bit exp, 10-bit fraction.
            Highest Tensor Core throughput, limited to ~65504 range.
            2 bytes/element.
            Use for: inference, mixed-precision training with loss scaling.
""")
