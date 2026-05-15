"""
Multi-restart wrapper around CDLNSPlacer.

Runs the inner placer N times with different seeds; picks the best result
by true proxy cost (computed via PlacementCost).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def _load_inner():
    path = Path(__file__).resolve().parent / "placer.py"
    spec = importlib.util.spec_from_file_location("cdlns_inner", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cdlns_inner"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plc_for(benchmark: Benchmark):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir

    name = benchmark.name
    iccad = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if iccad.exists():
        _, plc = load_benchmark_from_dir(str(iccad))
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
        "ariane133": "ariane133",
        "ariane136": "ariane136",
        "nvdla": "nvdla",
        "mempool_tile": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


class MultiStartPlacer:
    """Runs CDLNSPlacer with multiple seeds and returns the lowest-proxy-cost result."""

    def __init__(
        self,
        n_restarts: int = 5,
        per_run_seconds: float = 650.0,
        seeds: tuple = (42, 7, 2024, 1234, 31415),
        hill_climb_seconds: float = 0.0,
        true_proxy_check_every: float = 0.0,
        perturbations: tuple = (0.0, 0.0, 0.0, 0.0, 0.0),
        verbose: bool = True,
    ) -> None:
        self.n_restarts = n_restarts
        self.per_run_seconds = per_run_seconds
        self.seeds = seeds
        self.hill_climb_seconds = hill_climb_seconds
        self.true_proxy_check_every = true_proxy_check_every
        self.perturbations = perturbations
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        inner_mod = _load_inner()
        InnerCls = inner_mod.CDLNSPlacer

        # Load plc once for proxy evaluation between restarts.
        plc = _load_plc_for(benchmark)

        best_proxy = float("inf")
        best_pl: torch.Tensor = benchmark.macro_positions.clone()
        t0 = time.time()
        for k in range(self.n_restarts):
            seed = self.seeds[k % len(self.seeds)] if self.seeds else 42 + k
            p = (
                float(self.perturbations[k % len(self.perturbations)])
                if self.perturbations else 0.0
            )
            inner = InnerCls(
                seed=seed,
                max_seconds=self.per_run_seconds,
                cd_passes=30,
                ramp_steps=100,
                hill_climb_seconds=self.hill_climb_seconds,
                true_proxy_check_every=self.true_proxy_check_every,
                initial_perturbation=p,
                verbose=False,
            )
            if self.verbose:
                print(f"  [multi-start {k+1}/{self.n_restarts}] seed={seed} pert={p:.2f} t={time.time()-t0:.1f}s")
            placement = inner.place(benchmark)
            if plc is not None:
                costs = compute_proxy_cost(placement, benchmark, plc)
                proxy = float(costs["proxy_cost"])
                overlaps = int(costs["overlap_count"])
                if self.verbose:
                    print(
                        f"  [multi-start {k+1}] proxy={proxy:.4f} "
                        f"(wl={costs['wirelength_cost']:.3f} "
                        f"den={costs['density_cost']:.3f} "
                        f"cong={costs['congestion_cost']:.3f}) overlaps={overlaps}"
                    )
                if overlaps == 0 and proxy < best_proxy:
                    best_proxy = proxy
                    best_pl = placement.clone()
            else:
                # No plc available; just keep last result
                best_pl = placement.clone()
        if self.verbose and plc is not None:
            print(f"  [multi-start] best_proxy={best_proxy:.4f} total t={time.time()-t0:.1f}s")
        return best_pl
