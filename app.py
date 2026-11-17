import math
import statistics

import pandas as pd
import streamlit as st

from metrics import (
    bsgs_bytes_per_entry,
    bsgs_table_entries,
    calibrate_rho,
    fmt_bytes,
    fmt_count,
    fmt_seconds,
    SECONDS_PER_YEAR,
)
from plots import bar_chart, histogram, line_chart, to_png
from random_numbers_generator import generate_dlp_parameters, is_prime
from solvers import (
    BSGS,
    BSGS_SELF_TRACES_MEMORY,
    RHO,
    SOLVERS,
    aggregate_runs,
    is_success,
    rho_iteration_samples,
    run_measured,
)

try:  # the fixed generator exports this; fall back if you use an older one
    from random_numbers_generator import MIN_P_MARGIN
except ImportError:
    MIN_P_MARGIN = 4

# ======================================================================
# Settings you may want to tweak
# ======================================================================
USE_NATIVE_TABS = True  # False -> a radio-button navigation (see note in the README text below)
Q_BITS_MAX = 256        # generation works up to the assignment's target size
P_BITS_MAX = 1024
BENCH_BITS_MAX = 64

RHO_WARN_SECONDS = 60           # amber above this estimate
RHO_RED_SECONDS = 3600          # red above this estimate (still allowed, just very long)
BSGS_WARN_BYTES = 500 * 2**20   # amber above 500 MB
BSGS_BLOCK_BYTES = 2 * 2**30    # red AND blocked above 2 GB (a MemoryError can kill the app)

AGE_OF_UNIVERSE_YEARS = 1.38e10

st.set_page_config(page_title="DLP Lab: Pollard's Rho vs BSGS", layout="wide")

# ======================================================================
# Session state: everything the UI redraws from lives here
# ======================================================================
DEFAULTS = {
    "instance": None,          # dict: p, q, g, h, secret_x, bit sizes, checks
    "instance_counter": 0,
    "is_processing": False,    # True while a job is queued/running
    "job": None,               # "single" | "comparison" | "benchmark" | "analysis"
    "single_results": {},      # {algorithm: result}
    "comparison": None,        # {algorithm: result or skipped-dict}
    "benchmark": None,         # {"df": DataFrame, "algs": [...]}
    "analysis": None,          # {"counts": [...], "duds": n, "q": q, "q_bits": n}
}
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

busy = st.session_state.is_processing


def start_job(name):
    st.session_state.is_processing = True
    st.session_state.job = name
    st.rerun()


def end_job():
    st.session_state.is_processing = False
    st.session_state.job = None


# ======================================================================
# Shared helpers
# ======================================================================
@st.cache_data(show_spinner="Calibrating speed on this machine (one-off)...")
def calibrate():
    """(seconds per rho iteration, rho iterations per sqrt(q))."""
    return calibrate_rho()


def level_icon(level):
    return {"green": "🟢", "amber": "🟡", "red": "🔴"}[level]


def feasibility(q, p_bits):
    sec_per_iter, iters_per_sqrt_q = calibrate()
    rho_iters = iters_per_sqrt_q * math.sqrt(q)
    rho_secs = rho_iters * sec_per_iter
    entries = bsgs_table_entries(q)
    bsgs_bytes = entries * bsgs_bytes_per_entry(p_bits)

    if rho_secs < RHO_WARN_SECONDS:
        rho_level = "green"
    elif rho_secs < RHO_RED_SECONDS:
        rho_level = "amber"
    else:
        rho_level = "red"

    if bsgs_bytes < BSGS_WARN_BYTES:
        bsgs_level = "green"
    elif bsgs_bytes < BSGS_BLOCK_BYTES:
        bsgs_level = "amber"
    else:
        bsgs_level = "red"

    return {
        "rho_iters": rho_iters, "rho_secs": rho_secs, "rho_level": rho_level,
        "entries": entries, "bsgs_bytes": bsgs_bytes, "bsgs_level": bsgs_level,
    }


def bsgs_block_reason(feas):
    return (
        f"BSGS is blocked for this size: its table would need about "
        f"{fmt_bytes(feas['bsgs_bytes'])} (limit {fmt_bytes(BSGS_BLOCK_BYTES)}). "
        f"A MemoryError could crash the whole app."
    )


def validate_instance(inst):
    p, q, g, h, x = inst["p"], inst["q"], inst["g"], inst["h"], inst["secret_x"]
    return [
        ("p is prime", is_prime(p)),
        ("q is prime", is_prime(q)),
        ("p has the requested bit-length", p.bit_length() == inst["p_bits"]),
        ("q has the requested bit-length", q.bit_length() == inst["q_bits"]),
        ("q divides p − 1", (p - 1) % q == 0),
        ("g ≠ 1", g != 1),
        ("g^q ≡ 1 (mod p), so g has order q", pow(g, q, p) == 1),
        ("h^q ≡ 1 (mod p), so h is in the subgroup", pow(h, q, p) == 1),
        ("g^x ≡ h (mod p) for the stored secret x", pow(g, x, p) == h),
    ]


def make_instance(q_bits, p_bits):
    p, q, g, x, h = generate_dlp_parameters(int(q_bits), int(p_bits))
    inst = {"p": p, "q": q, "g": g, "h": h, "secret_x": x,
            "q_bits": int(q_bits), "p_bits": int(p_bits)}
    inst["checks"] = validate_instance(inst)
    inst["valid"] = all(ok for _, ok in inst["checks"])
    return inst


def render_result(res, inst, title=None):
    if res.get("skipped"):
        if title:
            st.markdown(f"### {title}")
        st.warning(res["reason"])
        return
    if title:
        st.markdown(f"### {title}")

    if res["k"] is None:
        st.error("The solver did not return a solution.")
    elif res["verified"] and res["matches_secret"]:
        st.success("Verified: g^k ≡ h (mod p) holds and k equals the secret exponent.")
    else:
        st.error("Verification FAILED: the returned k is not the secret exponent.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Recovered k", str(res["k"]))
    c2.metric("Actual secret x", str(inst["secret_x"]))
    c3.metric("g^k ≡ h (mod p)", "✅ yes" if res["verified"] else "❌ no")

    d1, d2, d3 = st.columns(3)
    d1.metric("Time", fmt_seconds(res["time_s"]))
    d2.metric("Peak memory", fmt_bytes(res["peak_bytes"]))
    d3.metric("Group multiplications", fmt_count(res["ops"]))

    labels = {"iterations": "iterations", "attempts": "attempts",
              "table_entries": "table entries", "giant_steps": "giant steps"}
    details = " · ".join(f"{labels.get(k, k)}: {fmt_count(v)}" for k, v in res["extra"].items())
    note = f" · median of {res['trials']} runs" if res.get("trials", 1) > 1 else ""
    st.caption(details + note)
    if res["peak_bytes"] is None:
        st.caption("Memory was not measured for this run.")


def png_download(fig, file_name, key):
    st.pyplot(fig)
    st.download_button("⬇ Download PNG", data=to_png(fig), file_name=file_name,
                       mime="image/png", key=key)


def p_bits_for(bits, scaling):
    target = {"p = 2 × q bits": 2 * bits, "Fixed 64-bit p": 64, "Fixed 128-bit p": 128}[scaling]
    return min(max(target, bits + MIN_P_MARGIN), P_BITS_MAX)


# ======================================================================
# Page 1: Instance
# ======================================================================
def page_instance():
    st.subheader("Generate a test instance")
    st.markdown(
        r"Builds $g^x \equiv h \pmod p$ where $g$ has prime order $q$ and $q \mid p-1$. "
        r"You choose only the two bit-lengths. Everything else is generated, and the "
        r"secret $x$ is kept aside purely to verify the solvers' answers."
    )

    c1, c2 = st.columns(2)
    with c1:
        q_bits = st.number_input("Subgroup bit-length (q)", min_value=8, max_value=Q_BITS_MAX,
                                 value=24, step=2, disabled=busy, key="inst_q_bits")
    min_p = int(q_bits) + MIN_P_MARGIN
    with c2:
        p_bits = st.number_input("Modulus bit-length (p)", min_value=min_p, max_value=P_BITS_MAX,
                                 value=min(max(48, min_p), P_BITS_MAX), step=2, disabled=busy)
    st.caption(
        f"p must be at least {MIN_P_MARGIN} bits longer than q (enforced). "
        f"The assignment target is q ≈ 256 bits: generating works, solving is infeasible."
    )

    if st.button("Generate instance", type="primary", disabled=busy, key="inst_generate"):
        with st.spinner("Generating primes and a generator..."):
            inst = make_instance(q_bits, p_bits)
        st.session_state.instance = inst
        st.session_state.instance_counter += 1
        st.session_state.single_results = {}
        st.session_state.comparison = None
        st.rerun()

    inst = st.session_state.instance
    if inst is None:
        st.info("No instance yet. Choose the bit-lengths and click **Generate instance**.")
        return

    st.divider()
    st.subheader("Public parameters (what the attacker sees)")
    left, right = st.columns(2)
    for col, items in ((left, [("Modulus p", "p"), ("Generator g", "g")]),
                       (right, [("Subgroup order q", "q"), ("Target h = g^x mod p", "h")])):
        with col:
            for label, key in items:
                st.markdown(f"**{label}** ({inst[key].bit_length()} bits)")
                st.code(str(inst[key]), language=None, wrap_lines=True)

    with st.expander("Secret exponent x (known only to the test harness)"):
        st.code(str(inst["secret_x"]), language=None, wrap_lines=True)

    if inst["valid"]:
        st.success("Instance is valid: all checks passed.")
    else:
        st.error("Instance failed validation. Generate a new one.")
    with st.expander("Validation checklist"):
        for label, ok in inst["checks"]:
            st.write(("✅ " if ok else "❌ ") + label)

    st.divider()
    st.subheader("Feasibility on this machine")
    feas = feasibility(inst["q"], inst["p_bits"])
    a, b = st.columns(2)
    with a:
        st.markdown(f"#### {level_icon(feas['rho_level'])} {RHO}")
        st.write(f"Expected iterations: {fmt_count(feas['rho_iters'])}")
        st.write(f"Estimated time: {fmt_seconds(feas['rho_secs'])}")
        if feas["rho_level"] == "amber":
            st.warning("This may take a while.")
        elif feas["rho_level"] == "red":
            st.error("Very long run. Allowed, but expect to wait a long time.")
    with b:
        st.markdown(f"#### {level_icon(feas['bsgs_level'])} {BSGS}")
        st.write(f"Baby-step table entries: {fmt_count(feas['entries'])}")
        st.write(f"Estimated peak memory: {fmt_bytes(feas['bsgs_bytes'])}")
        if feas["bsgs_level"] == "amber":
            st.warning("Large table. Close other programs first.")
        elif feas["bsgs_level"] == "red":
            st.error(bsgs_block_reason(feas))
    st.caption("Estimates come from a short calibration run on this machine; real values vary.")


# ======================================================================
# Page 2: Single attack
# ======================================================================
def page_single():
    inst = st.session_state.instance
    if inst is None:
        st.info("Generate an instance in the **Instance** tab first.")
        return

    st.subheader("Run one algorithm")
    alg = st.radio("Algorithm", list(SOLVERS), horizontal=True, disabled=busy, key="single_alg")
    options = {}
    if alg == RHO:
        options["max_retries"] = int(st.number_input(
            "Max retries after a dud collision", min_value=1, max_value=20, value=5,
            disabled=busy, key="single_retries"))
    measure_mem = st.checkbox("Also measure peak memory (runs the solver a second time)",
                              value=True, disabled=busy, key="single_mem")

    feas = feasibility(inst["q"], inst["p_bits"])
    blocked = alg == BSGS and feas["bsgs_level"] == "red"
    if blocked:
        st.error(bsgs_block_reason(feas))
    elif alg == RHO and feas["rho_level"] != "green":
        st.warning(f"Estimated rho time: {fmt_seconds(feas['rho_secs'])}.")
    elif alg == BSGS and feas["bsgs_level"] != "green":
        st.warning(f"Estimated BSGS memory: {fmt_bytes(feas['bsgs_bytes'])}.")

    if st.button("Run attack", type="primary", disabled=busy or blocked, key="single_run"):
        start_job("single")

    if st.session_state.job == "single":
        try:
            with st.spinner(f"Running {alg}..."):
                res = run_measured(alg, inst, measure_memory=measure_mem, **options)
            st.session_state.single_results[alg] = res
        finally:
            end_job()
        st.rerun()

    res = st.session_state.single_results.get(alg)
    if res is not None:
        st.divider()
        render_result(res, inst)


# ======================================================================
# Page 3: Comparison
# ======================================================================
def page_comparison():
    inst = st.session_state.instance
    if inst is None:
        st.info("Generate an instance in the **Instance** tab first.")
        return

    st.subheader("Compare both algorithms on the same instance")
    st.markdown("They run one after the other (not at the same time) on the identical "
                "instance, so the numbers are directly comparable.")

    c1, c2 = st.columns(2)
    with c1:
        trials = int(st.number_input("Rho runs (the median is reported)", min_value=1, max_value=25,
                                     value=5, disabled=busy, key="cmp_trials"))
    with c2:
        measure_mem = st.checkbox("Measure peak memory (extra traced run)", value=True,
                                  disabled=busy, key="cmp_mem")
    st.caption("Rho is randomized, so its time varies from run to run; BSGS is deterministic "
               "for a given instance.")

    feas = feasibility(inst["q"], inst["p_bits"])
    if feas["bsgs_level"] == "red":
        st.error(bsgs_block_reason(feas) + " It will be skipped; Rho still runs.")
    if feas["rho_level"] != "green":
        st.warning(f"Estimated rho time per run: {fmt_seconds(feas['rho_secs'])} (x{trials} runs).")

    if st.button("Run comparison", type="primary", disabled=busy, key="cmp_run"):
        start_job("comparison")

    if st.session_state.job == "comparison":
        try:
            out = {}
            with st.status("Running both algorithms on the same instance...", expanded=True) as status:
                for alg in (RHO, BSGS):
                    if alg == BSGS and feas["bsgs_level"] == "red":
                        out[alg] = {"skipped": True, "reason": bsgs_block_reason(feas)}
                        st.write(f"Skipping {alg} (memory limit).")
                        continue
                    n_runs = trials if alg == RHO else 1
                    st.write(f"Running {alg}" + (f" ({n_runs} runs)..." if n_runs > 1 else "..."))
                    runs = [
                        run_measured(alg, inst, measure_memory=(measure_mem and i == 0))
                        for i in range(n_runs)
                    ]
                    out[alg] = aggregate_runs(runs)
                status.update(label="Comparison complete", state="complete", expanded=False)
            st.session_state.comparison = out
        finally:
            end_job()
        st.rerun()

    comp = st.session_state.comparison
    if not comp:
        return

    st.divider()
    left, right = st.columns(2)
    with left:
        render_result(comp[RHO], inst, RHO)
    with right:
        render_result(comp[BSGS], inst, BSGS)

    done = [a for a in (RHO, BSGS) if not comp[a].get("skipped")]
    if len(done) == 2:
        r, b = comp[RHO], comp[BSGS]

        def ratio(x, y):
            return f"{y / x:.2f}×" if x and y is not None else "—"

        st.markdown("#### Side by side")
        table = pd.DataFrame(
            [
                ["Time", fmt_seconds(r["time_s"]), fmt_seconds(b["time_s"]), ratio(r["time_s"], b["time_s"])],
                ["Peak memory", fmt_bytes(r["peak_bytes"]), fmt_bytes(b["peak_bytes"]),
                 ratio(r["peak_bytes"], b["peak_bytes"]) if r["peak_bytes"] and b["peak_bytes"] else "—"],
                ["Group multiplications", fmt_count(r["ops"]), fmt_count(b["ops"]), ratio(r["ops"], b["ops"])],
            ],
            columns=["Metric", RHO, BSGS, "BSGS ÷ Rho"],
        )
        st.dataframe(table, hide_index=True)

    if done:
        st.markdown("#### Charts (log scale)")
        labels = done
        c1, c2, c3 = st.columns(3)
        with c1:
            png_download(bar_chart(labels, [comp[a]["time_s"] for a in labels],
                                   "Time", "seconds", fmt_seconds), "comparison_time.png", "cmp_png_time")
        with c2:
            png_download(bar_chart(labels, [comp[a]["peak_bytes"] for a in labels],
                                   "Peak memory", "bytes", fmt_bytes), "comparison_memory.png", "cmp_png_mem")
        with c3:
            png_download(bar_chart(labels, [comp[a]["ops"] for a in labels],
                                   "Group multiplications", "operations", fmt_count),
                         "comparison_ops.png", "cmp_png_ops")


# ======================================================================
# Page 4: Benchmark
# ======================================================================
def series_from_df(df, column, algs):
    bits = sorted(df["bits"].unique())
    series = {}
    for a in algs:
        sub = df[df["algorithm"] == a].set_index("bits")[column]
        series[a] = [None if (b not in sub.index or pd.isna(sub[b])) else float(sub[b]) for b in bits]
    return bits, series


def page_benchmark():
    st.subheader("Benchmark across subgroup sizes")
    st.markdown(r"Each size gets fresh random instances; both algorithms attack the **same** "
                r"instance within a trial. Expected: about $2\times$ work per $+2$ bits of $q$.")

    c1, c2, c3 = st.columns(3)
    with c1:
        min_bits = int(st.number_input("Smallest q (bits)", min_value=8, max_value=BENCH_BITS_MAX - 2,
                                       value=12, step=2, disabled=busy, key="bm_min"))
    with c2:
        max_bits = int(st.number_input("Largest q (bits)", min_value=min_bits + 2, max_value=BENCH_BITS_MAX,
                                       value=min(max(28, min_bits + 2), BENCH_BITS_MAX), step=2, disabled=busy))
    with c3:
        step = int(st.selectbox("Step (bits)", [2, 4, 6, 8], index=0, disabled=busy, key="bm_step"))

    c4, c5 = st.columns(2)
    with c4:
        trials = int(st.number_input("Trials per size (median is plotted)", min_value=1, max_value=25,
                                     value=3, disabled=busy, key="bm_trials"))
        algs = st.multiselect("Algorithms", list(SOLVERS), default=list(SOLVERS), disabled=busy, key="bm_algs")
    with c5:
        scaling = st.selectbox("Modulus (p) scaling", ["p = 2 × q bits", "Fixed 64-bit p", "Fixed 128-bit p"],
                               disabled=busy, key="bm_scaling")
        measure_mem = st.checkbox("Measure peak memory (extra traced run per size)", value=True,
                                  disabled=busy, key="bm_mem")
    st.caption(f"A fixed p is raised to q bits + {MIN_P_MARGIN} when q grows past it. "
               f"BSGS is skipped automatically at sizes above its memory limit.")

    sizes = list(range(min_bits, max_bits + 1, step))
    if sizes[-1] != max_bits:
        sizes.append(max_bits)

    sec_per_iter, iters_per_sqrt_q = calibrate()
    est_rho = sum(iters_per_sqrt_q * math.sqrt(2.0**b) * sec_per_iter * trials for b in sizes)
    if RHO in algs:
        st.write(f"Estimated rho time for this sweep: **{fmt_seconds(est_rho)}**"
                 + (" (about double with memory measurement on)" if measure_mem else ""))
        if est_rho > 300:
            st.warning("This sweep is long. Lower the largest size or the number of trials.")

    if st.button("Run benchmark", type="primary", disabled=busy or not algs, key="bm_run"):
        start_job("benchmark")

    if st.session_state.job == "benchmark":
        try:
            rows = []
            bar = st.progress(0.0, text="Starting benchmark...")
            for idx, bits in enumerate(sizes):
                bar.progress(idx / len(sizes), text=f"Testing {bits}-bit subgroup... ({idx + 1}/{len(sizes)})")
                pb = p_bits_for(bits, scaling)
                per_alg = {a: [] for a in algs}
                skipped = set()
                for t in range(trials):
                    inst = make_instance(bits, pb)
                    feas = feasibility(inst["q"], pb)
                    for a in algs:
                        if a == BSGS and feas["bsgs_level"] == "red":
                            skipped.add(a)
                            continue
                        per_alg[a].append(run_measured(a, inst, measure_memory=(measure_mem and t == 0)))
                for a in algs:
                    runs = per_alg[a]
                    good = [r for r in runs if is_success(r)]
                    rows.append({
                        "bits": bits, "p_bits": pb, "algorithm": a, "trials": trials,
                        "successes": len(good),
                        "median_time_s": statistics.median(r["time_s"] for r in good) if good else None,
                        "median_ops": statistics.median(r["ops"] for r in good) if good else None,
                        "peak_memory_bytes": next((r["peak_bytes"] for r in runs if r["peak_bytes"] is not None), None),
                        "note": "skipped: memory limit" if a in skipped else "",
                    })
            bar.progress(1.0, text="Benchmark complete!")
            st.session_state.benchmark = {"df": pd.DataFrame(rows), "algs": list(algs)}
        finally:
            end_job()
        st.rerun()

    bench = st.session_state.benchmark
    if not bench:
        return
    df, used = bench["df"], bench["algs"]

    st.divider()
    st.subheader("Results (log scale)")
    charts = [
        ("median_time_s", "Time vs subgroup size", "Median time (s)", "benchmark_time.png"),
        ("peak_memory_bytes", "Peak memory vs subgroup size", "Peak memory (bytes)", "benchmark_memory.png"),
        ("median_ops", "Group multiplications vs subgroup size", "Median group multiplications",
         "benchmark_ops.png"),
    ]
    for column, title, ylabel, file_name in charts:
        bits, series = series_from_df(df, column, used)
        if all(v is None for ys in series.values() for v in ys):
            st.info(f"No data for “{title}”" + (" (memory measurement was off)." if "memory" in column else "."))
            continue
        png_download(line_chart(bits, series, title, "Subgroup size (bits of q)", ylabel),
                     file_name, f"bm_png_{column}")

    failed = df[(df["successes"] < df["trials"]) | (df["note"] != "")]
    if len(failed):
        st.warning("Some runs failed or were skipped. They are excluded from the medians; see the table below.")
    with st.expander("Raw data"):
        st.dataframe(df, hide_index=True)
    st.download_button("⬇ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="benchmark.csv", mime="text/csv", key="bm_csv")


# ======================================================================
# Page 5: 256-bit analysis (task e)
# ======================================================================
def page_analysis():
    st.subheader("Expected cost at the target size")
    st.markdown(
        r"Rho walks inside the order-$q$ subgroup, so by the birthday effect it needs on the order of "
        r"$\sqrt{\pi q/2}$ steps; BSGS needs a table of about $\sqrt q$ entries. Only "
        r"$q$ matters for the step count, not $p$."
    )

    c1, c2 = st.columns(2)
    with c1:
        target_bits = int(st.number_input("Target subgroup size (bits of q)", min_value=16, max_value=512,
                                          value=256, disabled=busy, key="an_bits"))
    with c2:
        target_p = int(st.number_input("Modulus size (bits of p), for BSGS memory",
                                       min_value=target_bits + MIN_P_MARGIN, max_value=16384,
                                       value=max(3072, target_bits + MIN_P_MARGIN), step=64, disabled=busy))

    sec_per_iter, iters_per_sqrt_q = calibrate()
    q = 2**target_bits
    iters = iters_per_sqrt_q * math.sqrt(q)
    seconds = iters * sec_per_iter
    years = seconds / SECONDS_PER_YEAR
    entries = math.sqrt(q)
    table_bytes = entries * bsgs_bytes_per_entry(target_p)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Pollard's Rho")
        st.metric("Expected iterations", f"≈ 2^{math.log2(iters):.1f}")
        st.metric("Extrapolated time", fmt_seconds(seconds))
        if years > 1:
            st.caption(f"That is about {years / AGE_OF_UNIVERSE_YEARS:.2e}× the age of the universe, "
                       f"at {fmt_seconds(sec_per_iter)} per iteration measured here.")
        st.caption("Memory stays tiny: three (y, a, b) states.")
    with b:
        st.markdown("#### Baby-step Giant-step")
        st.metric("Table entries", f"≈ 2^{math.log2(entries):.1f}")
        st.metric("Estimated memory", fmt_bytes(table_bytes))
        st.caption("Memory, not time, is what stops BSGS first.")

    st.divider()
    st.subheader("Rho is randomized: the spread of its running time")
    st.markdown(
        "The walk's path depends on the starting point, so the iteration count differs from run to run "
        "even on the same instance. This runs rho many times with random starts on one instance and "
        "plots how often each count occurs."
    )
    c3, c4 = st.columns(2)
    with c3:
        an_q_bits = int(st.number_input("Subgroup size for the experiment (bits)", min_value=12, max_value=28,
                                        value=20, step=2, disabled=busy, key="an_q"))
    with c4:
        runs = int(st.number_input("Number of runs", min_value=20, max_value=1000, value=200, step=20,
                                   disabled=busy, key="an_runs"))

    if st.button("Run experiment", type="primary", disabled=busy, key="an_run"):
        start_job("analysis")

    if st.session_state.job == "analysis":
        try:
            with st.spinner(f"Running rho {runs} times..."):
                inst = make_instance(an_q_bits, 2 * an_q_bits)
                counts, duds = rho_iteration_samples(inst, runs)
            st.session_state.analysis = {"counts": counts, "duds": duds,
                                         "q": inst["q"], "q_bits": an_q_bits}
        finally:
            end_job()
        st.rerun()

    res = st.session_state.analysis
    if not res or not res["counts"]:
        return
    counts = res["counts"]
    theory = math.sqrt(math.pi * res["q"] / 2)
    mean = statistics.mean(counts)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean iterations", fmt_count(mean))
    m2.metric("Median", fmt_count(statistics.median(counts)))
    m3.metric("Min / max", f"{min(counts):,} / {max(counts):,}")
    m4.metric("Duds", f"{res['duds']} of {len(counts) + res['duds']}")
    st.caption(f"Birthday-effect scale √(πq/2) ≈ {fmt_count(theory)}. Floyd's cycle detection "
               f"typically needs a small constant factor more, so the mean lands near, not on, that line.")

    fig = histogram(counts, f"Rho iterations over {len(counts)} runs (q = {res['q_bits']} bits)",
                    "Iterations until collision",
                    vlines=[(theory, "theory √(πq/2)"), (mean, "measured mean")])
    png_download(fig, "rho_randomness.png", "an_png_hist")


# ======================================================================
# Layout: status strip + tabs (no sidebar)
# ======================================================================
st.title("Discrete Logarithm Lab: Pollard's Rho vs BSGS")

if BSGS_SELF_TRACES_MEMORY:
    st.warning("bsgs.py starts tracemalloc itself (no `track_memory` flag). BSGS memory will read 0 and its "
               "time will be inflated. Add the flag as in the shared bsgs.py.")

_inst = st.session_state.instance
_strip, _state = st.columns([5, 1])
with _strip:
    if _inst is None:
        st.caption("No instance yet. Open the **Instance** tab to generate one.")
    else:
        ok = "✅ valid" if _inst["valid"] else "❌ invalid"
        st.caption(f"Instance #{st.session_state.instance_counter}: q = {_inst['q_bits']} bits · "
                   f"p = {_inst['p_bits']} bits · {ok}")
with _state:
    st.caption("⏳ Running..." if busy else "Idle")

PAGES = [
    ("1 · Instance", page_instance),
    ("2 · Single Attack", page_single),
    ("3 · Comparison", page_comparison),
    ("4 · Benchmark", page_benchmark),
    ("5 · 256-bit Analysis", page_analysis),
]

if USE_NATIVE_TABS:
    for _tab, (_name, _page) in zip(st.tabs([n for n, _ in PAGES]), PAGES):
        with _tab:
            _page()
else:
    # Fallback if st.tabs jumps back to the first tab after a rerun in your version.
    _choice = st.radio("Section", [n for n, _ in PAGES], horizontal=True, key="page",
                       label_visibility="collapsed", disabled=busy)
    dict(PAGES)[_choice]()
