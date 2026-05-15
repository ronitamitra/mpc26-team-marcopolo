"""Single-seed deep-simmer ibm01: 650s budget, vanilla config, seed=42."""
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
    max_s = float(sys.argv[2]) if len(sys.argv) > 2 else 650.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    bench_path = ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(bench_path))
    print(f"loaded {bench}: nh={benchmark.num_hard_macros} ns={benchmark.num_soft_macros}", flush=True)

    mod = _load()
    placer = mod.CDLNSPlacer(
        seed=seed,
        max_seconds=max_s,
        # Vanilla config — all gates off:
        incremental_init=False,
        penalty_ramp=False,
        annealed_cd=False,
        verbose=False,
    )
    print(f"running vanilla CDLNS seed={seed} budget={max_s}s ...", flush=True)
    t0 = time.time()
    placement = placer.place(benchmark)
    elapsed = time.time() - t0

    costs = compute_proxy_cost(placement, benchmark, plc)
    proxy = float(costs["proxy_cost"])
    overlaps = int(costs["overlap_count"])
    wl = float(costs["wirelength_cost"])
    den = float(costs["density_cost"])
    cong = float(costs["congestion_cost"])
    legality = "LEGAL" if overlaps == 0 else f"ILLEGAL ({overlaps} overlaps)"

    print(f"\nRESULT: proxy={proxy:.4f}  (wl={wl:.3f}  den={den:.3f}  cong={cong:.3f})"
          f"  {legality}  elapsed={elapsed:.1f}s", flush=True)

    prev = 1.0717
    delta = proxy - prev
    pct = 100.0 * delta / prev
    print(f"vs prior multistart-winner ({prev:.4f}): delta={delta:+.4f} ({pct:+.2f}%)", flush=True)
    print(f"VERDICT: {'BEATS 1.0717 — launch full sweep' if delta < -1e-4 and overlaps == 0 else 'does not beat 1.0717'}", flush=True)


if __name__ == "__main__":
    main()
