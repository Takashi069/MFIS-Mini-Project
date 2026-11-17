"""Measurement, estimation and formatting helpers. No Streamlit in here."""
import math
import statistics
import sys
import time
import tracemalloc

SECONDS_PER_YEAR = 365.25 * 86400


# ---------------------------------------------------------------- measuring
def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_seconds). Nothing else is timed."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def peak_memory(fn, *args, **kwargs):
    """Run fn under tracemalloc and return (result, peak_bytes).

    tracemalloc slows code down, so never use this run for timing.
    """
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        result = fn(*args, **kwargs)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return result, max(0, peak - baseline)


# --------------------------------------------------------------- estimating
def expected_rho_iterations(q, iters_per_sqrt_q):
    """Floyd-loop iterations: calibrated constant * sqrt(q)."""
    return iters_per_sqrt_q * math.sqrt(q)


def bsgs_table_entries(q):
    """m = ceil(sqrt(q)) baby steps."""
    return math.isqrt(q - 1) + 1


def bsgs_bytes_per_entry(p_bits):
    """Estimated peak bytes per baby-step dict entry.

    value int (28) + key int (depends on p) + dict slot (~48), with a 15%
    margin for the resize copy that tracemalloc sees at the peak. Checked
    against tracemalloc: within about 15% for 56-, 128- and 512-bit p.
    """
    key_bytes = sys.getsizeof(1 << (p_bits - 1))
    return math.ceil(1.15 * (28 + key_bytes + 48))


def bsgs_estimated_bytes(q, p_bits):
    return bsgs_table_entries(q) * bsgs_bytes_per_entry(p_bits)


def calibrate_rho(runs=25, q_bits=20):
    """Measure this machine: (seconds per iteration, iterations per sqrt(q)).

    Uses medians over several small random instances (cheap, about a second).
    """
    # imported here so this module stays importable on its own
    from pollards_rho import pollards_rho_dlp
    from random_numbers_generator import generate_dlp_parameters

    sec_per_iter, ratios = [], []
    for _ in range(runs):
        p, q, g, _x, h = generate_dlp_parameters(q_bits, 2 * q_bits)
        (k, iters), elapsed = timed(pollards_rho_dlp, p, q, g, h)
        if k is None or iters == 0:
            continue
        sec_per_iter.append(elapsed / iters)
        ratios.append(iters / math.sqrt(q))
    if not ratios:  # extremely unlikely; fall back to sane defaults
        return 2e-6, 1.25
    return statistics.median(sec_per_iter), statistics.median(ratios)


# --------------------------------------------------------------- formatting
def fmt_seconds(s):
    if s is None:
        return "—"
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    if s < 60:
        return f"{s:.2f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    if s < 86400:
        return f"{s / 3600:.1f} h"
    if s < 86400 * 365:
        return f"{s / 86400:.1f} days"
    years = s / SECONDS_PER_YEAR
    if years < 1e6:
        return f"{years:,.0f} years"
    return f"{years:.2e} years (≈ 2^{math.log2(years):.0f})"


def fmt_bytes(n):
    if n is None:
        return "—"
    if n >= 2**90:
        return f"≈ 2^{math.log2(n):.0f} bytes"
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    x, i = float(n), 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:,.0f} {units[i]}" if i == 0 else f"{x:,.1f} {units[i]}"


def fmt_count(x):
    if x is None:
        return "—"
    if x < 1000:
        return f"{x:,.0f}"
    if x < 1e15:
        return f"{x:,.0f} (≈ 2^{math.log2(x):.1f})"
    return f"≈ 2^{math.log2(x):.1f}"
