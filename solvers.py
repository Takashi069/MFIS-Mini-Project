"""The one place where the UI meets the algorithms.

Every solver here has the same shape (adapters around the algorithm files):

    solver(p, q, g, h, **options) -> {"k": int | None,
                                      "ops": int,      # group multiplications
                                      "extra": dict}   # algorithm-specific numbers

Time and peak memory are measured OUTSIDE the solvers (run_measured), so the
algorithm files never have to measure themselves. No Streamlit in here.
"""
import inspect
import statistics

from bsgs import bsgs_dlp
from metrics import bsgs_table_entries, peak_memory, timed
from pollards_rho import pollards_rho_dlp

RHO = "Pollard's Rho"
BSGS = "Baby-step Giant-step"


def solve_rho(p, q, g, h, max_retries=5, **_ignored):
    total_iters = 0
    for attempt in range(max_retries):
        k, iters = pollards_rho_dlp(p, q, g, h, randomize=(attempt > 0))
        total_iters += iters
        if k is not None:
            break
    # One iteration = 1 tortoise step + 2 hare steps = 3 group multiplications.
    return {
        "k": k,
        "ops": 3 * total_iters,
        "extra": {"iterations": total_iters, "attempts": attempt + 1},
    }


# Partner's bsgs_dlp(p, q, g, h, track_memory=True) -> (x, iterations, memory_bytes).
# The app measures time and memory itself (same way for both algorithms), and
# tracemalloc inside the solver would zero the memory reading and inflate the
# timing about 16x, so it is switched off here. If a version of bsgs.py without
# the track_memory flag is dropped in, the app still runs but shows a warning.
_HAS_TRACK_FLAG = "track_memory" in inspect.signature(bsgs_dlp).parameters
BSGS_SELF_TRACES_MEMORY = not _HAS_TRACK_FLAG


def solve_bsgs(p, q, g, h, **_ignored):
    kwargs = {"track_memory": False} if _HAS_TRACK_FLAG else {}
    k, iterations, _own_memory = bsgs_dlp(p, q, g, h, **kwargs)
    table_size = bsgs_table_entries(q)  # m = ceil(sqrt(q)) baby steps
    # iterations = m baby steps + giant steps taken, one multiplication each.
    return {
        "k": k,
        "ops": iterations,
        "extra": {"table_entries": table_size, "giant_steps": max(iterations - table_size, 0)},
    }


SOLVERS = {RHO: solve_rho, BSGS: solve_bsgs}


def run_measured(name, inst, measure_memory=True, **options):
    """Run one solver on an instance and return a full, verified result.

    The solver only ever receives the PUBLIC values (p, q, g, h). The secret
    exponent is used afterwards, for verification only.
    """
    solver = SOLVERS[name]
    args = (inst["p"], inst["q"], inst["g"], inst["h"])

    res, elapsed = timed(solver, *args, **options)  # clean timing, no tracing

    peak = None
    if measure_memory:  # separate traced run: tracemalloc would distort the time
        _, peak = peak_memory(solver, *args, **options)

    k = res["k"]
    return {
        "algorithm": name,
        "k": k,
        "ops": res["ops"],
        "extra": res["extra"],
        "time_s": elapsed,
        "peak_bytes": peak,
        "verified": k is not None and pow(inst["g"], k, inst["p"]) == inst["h"],
        "matches_secret": k == inst["secret_x"],
        "trials": 1,
    }


def is_success(res):
    return bool(res["verified"] and res["matches_secret"])


def aggregate_runs(runs):
    """Combine several runs of the same algorithm on the same instance."""
    first = runs[0]
    extra = {
        key: statistics.median(r["extra"][key] for r in runs)
        for key in first["extra"]
    }
    return {
        "algorithm": first["algorithm"],
        "k": first["k"],
        "ops": statistics.median(r["ops"] for r in runs),
        "extra": extra,
        "time_s": statistics.median(r["time_s"] for r in runs),
        "peak_bytes": next((r["peak_bytes"] for r in runs if r["peak_bytes"] is not None), None),
        "verified": all(r["verified"] for r in runs),
        "matches_secret": all(r["matches_secret"] for r in runs),
        "trials": len(runs),
    }


def rho_iteration_samples(inst, runs):
    """Run rho many times on ONE instance with random starting points.

    Returns (iteration_counts, duds). Used to show rho's randomized behavior.
    """
    p, q, g, h = inst["p"], inst["q"], inst["g"], inst["h"]
    counts, duds = [], 0
    for _ in range(runs):
        k, iters = pollards_rho_dlp(p, q, g, h, randomize=True)
        if k is None:
            duds += 1
        else:
            counts.append(iters)
    return counts, duds
