"""
3-benchmark sanity sweep: ms5 baseline vs perturbed-restart ms5.

Perturbed config keeps seed 42 at perturbation=0 (safety baseline) and
escalates the other four seeds to explore different placement basins:
(0.0, 0.1, 0.2, 0.3, 0.5).
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

BENCHES = ["ibm01", "ibm08", "ibm15"]
MS5_BASELINE = {"ibm01": 1.1083, "ibm08": 1.4702, "ibm15": 1.6200}

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
        perturbations=(0.0, 0.1, 0.2, 0.3, 0.5),
        hill_climb_seconds=0.0,
        true_proxy_check_every=0.0,
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

print(f"\n{'='*60}\nSUMMARY")
print(f"{'bench':<8} {'pert':>8} {'ms5':>8} {'delta':>8} {'time':>8}  status")
ok_all = True
for bench, proxy, overlaps, elapsed, over, delta in results:
    status = "OVER" if over else ("OVERLAP" if overlaps else "OK")
    if over or overlaps:
        ok_all = False
    print(f"{bench:<8} {proxy:>8.4f} {MS5_BASELINE[bench]:>8.4f} {delta:>+8.4f} {elapsed:>7.1f}s  {status}")
avg_p = sum(r[1] for r in results) / len(results)
avg_b = sum(MS5_BASELINE[b] for b in BENCHES) / len(BENCHES)
print(f"{'AVG':<8} {avg_p:>8.4f} {avg_b:>8.4f} {avg_p-avg_b:>+8.4f}")
print(f"\nTotal time: {time.time() - total_t0:.1f}s")
print(f"VERDICT: {'perturbed_ms5 BEATS ms5' if ok_all and avg_p < avg_b - 1e-4 else 'no improvement vs ms5'}")
