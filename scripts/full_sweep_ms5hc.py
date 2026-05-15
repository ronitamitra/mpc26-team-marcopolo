"""
Full 17-benchmark ms5hc sweep.

Runs MultiStartPlacer on every IBM benchmark with HC enabled
(hill_climb_seconds=12, true_proxy_check_every=12). Prints per-benchmark
results and a final summary table comparing to the committed ms5 baseline.

Submission file (multistart_placer.py) is left untouched — this is a
test runner that constructs MultiStartPlacer directly with the HC config.
"""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

ms_path = ROOT / "submissions" / "cd_lns" / "multistart_placer.py"
spec = importlib.util.spec_from_file_location("ms", str(ms_path))
ms_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms_mod)

BENCHES = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17",
    "ibm18",
]

# ms5 baseline numbers (avg = 1.4960 over 17 benches)
MS5_BASELINE = {
    "ibm01": 1.1083, "ibm02": 1.5707, "ibm03": 1.3823, "ibm04": 1.3925,
    "ibm06": 1.6836, "ibm07": 1.5091, "ibm08": 1.4702, "ibm09": 1.1161,
    "ibm10": 1.4734, "ibm11": 1.2681, "ibm12": 1.6646, "ibm13": 1.4109,
    "ibm14": 1.6435, "ibm15": 1.6200, "ibm16": 1.5750, "ibm17": 1.7432,
    "ibm18": 1.7874,
}

results = []
total_t0 = time.time()
for bench in BENCHES:
    print(f"\n{'='*60}\n[{bench}] starting", flush=True)
    bench_path = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(bench_path))

    placer = ms_mod.MultiStartPlacer(
        n_restarts=5,
        per_run_seconds=200.0,
        seeds=(42, 7, 2024, 1234, 31415),
        hill_climb_seconds=12.0,
        true_proxy_check_every=12.0,
        verbose=True,
    )
    t0 = time.time()
    placement = placer.place(benchmark)
    elapsed = time.time() - t0

    costs = compute_proxy_cost(placement, benchmark, plc)
    proxy = float(costs["proxy_cost"])
    overlaps = int(costs["overlap_count"])
    over_budget = elapsed > 3600.0
    delta = proxy - MS5_BASELINE[bench]
    results.append((bench, proxy, overlaps, elapsed, over_budget, delta))
    print(
        f"[{bench}] proxy={proxy:.4f} (vs ms5 {MS5_BASELINE[bench]:.4f}, "
        f"delta={delta:+.4f}) overlaps={overlaps} "
        f"elapsed={elapsed:.1f}s {'OVER BUDGET' if over_budget else 'ok'}",
        flush=True,
    )

print(f"\n{'='*60}\nFINAL SUMMARY")
print(f"{'bench':<8} {'ms5hc':>8} {'ms5':>8} {'delta':>9} {'time':>8}  status")
print("-" * 60)
n_over = 0
n_overlap = 0
for bench, proxy, overlaps, elapsed, over, delta in results:
    if over:
        n_over += 1
    if overlaps:
        n_overlap += 1
    status = "OVER" if over else ("OVERLAP" if overlaps else "OK")
    print(f"{bench:<8} {proxy:>8.4f} {MS5_BASELINE[bench]:>8.4f} {delta:>+9.4f} {elapsed:>7.1f}s  {status}")
avg_proxy = sum(r[1] for r in results) / len(results)
avg_baseline = sum(MS5_BASELINE[b] for b in BENCHES) / len(BENCHES)
print("-" * 60)
print(f"{'AVG':<8} {avg_proxy:>8.4f} {avg_baseline:>8.4f} {avg_proxy-avg_baseline:>+9.4f}")
print(f"\nTotal sweep time: {time.time() - total_t0:.1f}s")
print(f"Over-budget benchmarks: {n_over}/{len(results)}")
print(f"Benchmarks with overlaps: {n_overlap}/{len(results)}")
viable = (n_over == 0 and n_overlap == 0 and avg_proxy < avg_baseline)
print(f"VERDICT: {'ms5hc REPLACES ms5' if viable else 'KEEP ms5'}")
