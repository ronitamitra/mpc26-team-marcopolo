"""A/B test: penalty_ramp ON vs OFF on a given benchmark.

Fails loudly if either run produces overlaps — proxy comparisons are
meaningless if one side is illegal.
"""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402


def _load():
    path = ROOT / "submissions" / "cd_lns" / "placer.py"
    spec = importlib.util.spec_from_file_location("cdlns_inner", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    bench = sys.argv[1] if len(sys.argv) > 1 else "ibm01"
    max_s = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    bench_path = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, _plc = load_benchmark_from_dir(str(bench_path))
    print(f"loaded {bench}: nh={benchmark.num_hard_macros} ns={benchmark.num_soft_macros}", flush=True)
    mod = _load()

    configs = [
        ("vanilla_ms5", dict(incremental_init=False, penalty_ramp=False, annealed_cd=False)),
        ("annealed_cd", dict(incremental_init=False, penalty_ramp=False, annealed_cd=True)),
    ]
    results = {}
    for name, cfg in configs:
        _, plc_run = load_benchmark_from_dir(str(bench_path))
        placer = mod.CDLNSPlacer(
            seed=seed, max_seconds=max_s, verbose=False, **cfg,
        )
        t0 = time.time()
        placement = placer.place(benchmark)
        elapsed = time.time() - t0
        costs = compute_proxy_cost(placement, benchmark, plc_run)
        proxy = float(costs["proxy_cost"])
        overlaps = int(costs["overlap_count"])
        results[name] = dict(proxy=proxy, overlaps=overlaps,
                             wl=float(costs["wirelength_cost"]),
                             den=float(costs["density_cost"]),
                             cong=float(costs["congestion_cost"]),
                             elapsed=elapsed)
        legality = "LEGAL" if overlaps == 0 else f"ILLEGAL ({overlaps} overlaps)"
        print(f"{name}: proxy={proxy:.4f} (wl={results[name]['wl']:.3f} "
              f"den={results[name]['den']:.3f} cong={results[name]['cong']:.3f}) "
              f"{legality} elapsed={elapsed:.1f}s", flush=True)

    base = results["vanilla_ms5"]
    ramp = results["annealed_cd"]
    if base["overlaps"] > 0:
        print(f"\nABORT: vanilla_ms5 has {base['overlaps']} overlaps — placer is broken.", flush=True)
        sys.exit(2)
    if ramp["overlaps"] > 0:
        print(f"\nannealed_cd INVALID: {ramp['overlaps']} overlaps. baseline proxy={base['proxy']:.4f}.", flush=True)
        sys.exit(1)
    delta = ramp["proxy"] - base["proxy"]
    pct = 100.0 * delta / base["proxy"] if base["proxy"] != 0 else 0.0
    print(f"\nDELTA (annealed_cd - vanilla_ms5): {delta:+.4f} ({pct:+.2f}%)", flush=True)
    print(f"VERDICT: {'annealed_cd IMPROVES' if delta < -1e-4 else 'annealed_cd does not help (noise or worse)'}", flush=True)


if __name__ == "__main__":
    main()
