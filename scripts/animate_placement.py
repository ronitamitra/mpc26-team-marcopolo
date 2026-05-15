"""
Animate the CD+SA placer on a benchmark.

Two-panel layout matching the README's force-directed visual:
  - Left: macros + net "spider lines" + info box (step, cost, best, max disp)
  - Right: cost convergence curve

Output: an mp4 (preferred) or gif written to vis/<benchmark>_anim.<ext>.

Usage:
    uv run python scripts/animate_placement.py -b ibm01 --steps 150 --fps 12
    uv run python scripts/animate_placement.py -b ibm03 --gif
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.benchmark import Benchmark


# ── Inline CD step (mirrors submissions/cd_lns/placer.py logic) ────────────


class _CDPlacer:
    """A minimal CD pass implementation that yields snapshots after each pass."""

    def __init__(self, benchmark: Benchmark, plc, gap: float = 0.05, density_lambda: float = 30.0):
        self.bench = benchmark
        self.plc = plc
        self.gap = gap
        self.lam = density_lambda
        self.n = benchmark.num_macros
        self.nh = benchmark.num_hard_macros
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.pos = benchmark.macro_positions.numpy().astype(np.float64).copy()
        self.sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        self.movable = benchmark.get_movable_mask().numpy()
        self.hw = self.sizes[:, 0] / 2.0
        self.hh = self.sizes[:, 1] / 2.0

        # Per-net pin structure
        self.port_pos = (
            benchmark.port_positions.numpy().astype(np.float64)
            if benchmark.port_positions.shape[0] > 0
            else np.zeros((0, 2))
        )
        offsets_per_macro: List[np.ndarray] = []
        for i in range(self.nh):
            o = (
                benchmark.macro_pin_offsets[i].numpy().astype(np.float64)
                if i < len(benchmark.macro_pin_offsets)
                else np.zeros((0, 2))
            )
            offsets_per_macro.append(o)

        self.n_nets = len(benchmark.net_pin_nodes)
        self.net_owners: List[np.ndarray] = []
        self.net_ox: List[np.ndarray] = []
        self.net_oy: List[np.ndarray] = []
        self.net_is_port: List[np.ndarray] = []
        for nid in range(self.n_nets):
            entries = benchmark.net_pin_nodes[nid].numpy()
            owners = entries[:, 0].astype(np.int64)
            slots = entries[:, 1].astype(np.int64)
            kp = len(owners)
            ox = np.zeros(kp)
            oy = np.zeros(kp)
            is_port = np.zeros(kp, dtype=bool)
            for k in range(kp):
                o = int(owners[k])
                s = int(slots[k])
                if o < self.nh:
                    om = offsets_per_macro[o]
                    if om.shape[0] > s:
                        ox[k] = om[s, 0]
                        oy[k] = om[s, 1]
                elif o < self.n:
                    pass
                else:
                    pidx = o - self.n
                    is_port[k] = True
                    ox[k] = self.port_pos[pidx, 0]
                    oy[k] = self.port_pos[pidx, 1]
            self.net_owners.append(owners)
            self.net_ox.append(ox)
            self.net_oy.append(oy)
            self.net_is_port.append(is_port)

        # macro -> [(net_idx, pin_idx)]
        self.m_pins: List[List[Tuple[int, int]]] = [[] for _ in range(self.n)]
        for nid in range(self.n_nets):
            owners = self.net_owners[nid]
            isp = self.net_is_port[nid]
            for k in range(len(owners)):
                if not isp[k]:
                    o = int(owners[k])
                    if o < self.n:
                        self.m_pins[o].append((nid, k))

        self.pin_x: List[np.ndarray] = [None] * self.n_nets  # type: ignore[list-item]
        self.pin_y: List[np.ndarray] = [None] * self.n_nets  # type: ignore[list-item]
        self.net_minx = np.zeros(self.n_nets)
        self.net_maxx = np.zeros(self.n_nets)
        self.net_miny = np.zeros(self.n_nets)
        self.net_maxy = np.zeros(self.n_nets)
        self._recompute_pins()

        # Density grid (from placer.py, simplified)
        self.gr = int(benchmark.grid_rows)
        self.gc = int(benchmark.grid_cols)
        self.cell_w = self.cw / self.gc
        self.cell_h = self.ch / self.gr
        self.cell_area = self.cell_w * self.cell_h
        self.grid = np.zeros((self.gr, self.gc), dtype=np.float64)
        for mi in range(self.n):
            for r, c, a in self._cells_for(mi, float(self.pos[mi, 0]), float(self.pos[mi, 1])):
                self.grid[r, c] += a

        self.movable_idx = [i for i in range(self.nh) if self.movable[i]]
        self.last_max_disp = 0.0
        self.last_n_acc = 0

    # ------------------------------------------------------------------
    def _recompute_pins(self) -> None:
        for nid in range(self.n_nets):
            owners = self.net_owners[nid]
            ox = self.net_ox[nid]
            oy = self.net_oy[nid]
            isp = self.net_is_port[nid]
            kp = len(owners)
            if kp == 0:
                self.pin_x[nid] = np.zeros(0)
                self.pin_y[nid] = np.zeros(0)
                continue
            px = np.empty(kp)
            py = np.empty(kp)
            for k in range(kp):
                if isp[k]:
                    px[k] = ox[k]
                    py[k] = oy[k]
                else:
                    o = int(owners[k])
                    px[k] = self.pos[o, 0] + ox[k]
                    py[k] = self.pos[o, 1] + oy[k]
            self.pin_x[nid] = px
            self.pin_y[nid] = py
            self.net_minx[nid] = px.min()
            self.net_maxx[nid] = px.max()
            self.net_miny[nid] = py.min()
            self.net_maxy[nid] = py.max()

    def _cells_for(self, m: int, x: float, y: float) -> List[Tuple[int, int, float]]:
        xmin = x - self.hw[m]
        xmax = x + self.hw[m]
        ymin = y - self.hh[m]
        ymax = y + self.hh[m]
        c0 = max(0, int(xmin / self.cell_w))
        c1 = min(self.gc - 1, int(xmax / self.cell_w))
        r0 = max(0, int(ymin / self.cell_h))
        r1 = min(self.gr - 1, int(ymax / self.cell_h))
        out: List[Tuple[int, int, float]] = []
        for r in range(r0, r1 + 1):
            cy0 = r * self.cell_h
            cy1 = (r + 1) * self.cell_h
            y_ov = min(ymax, cy1) - max(ymin, cy0)
            if y_ov <= 0:
                continue
            for c in range(c0, c1 + 1):
                cx0 = c * self.cell_w
                cx1 = (c + 1) * self.cell_w
                x_ov = min(xmax, cx1) - max(xmin, cx0)
                if x_ov <= 0:
                    continue
                out.append((r, c, x_ov * y_ov))
        return out

    def _overlaps(self, m: int, cx: float, cy: float) -> bool:
        dx = np.abs(cx - self.pos[: self.nh, 0])
        dy = np.abs(cy - self.pos[: self.nh, 1])
        sx = (self.sizes[m, 0] + self.sizes[: self.nh, 0]) / 2.0 + self.gap
        sy = (self.sizes[m, 1] + self.sizes[: self.nh, 1]) / 2.0 + self.gap
        ov = (dx < sx) & (dy < sy)
        ov[m] = False
        return bool(ov.any())

    def _compute_target(self, m: int) -> Tuple[float, float]:
        sx = sy = 0.0
        cnt = 0
        for nid, k in self.m_pins[m]:
            if len(self.pin_x[nid]) <= 1:
                continue
            cx = (self.net_minx[nid] + self.net_maxx[nid]) / 2.0
            cy = (self.net_miny[nid] + self.net_maxy[nid]) / 2.0
            sx += cx - self.net_ox[nid][k]
            sy += cy - self.net_oy[nid][k]
            cnt += 1
        if cnt == 0:
            return float(self.pos[m, 0]), float(self.pos[m, 1])
        return sx / cnt, sy / cnt

    def _find_legal(self, m: int, tx: float, ty: float, max_rings: int = 60) -> Optional[Tuple[float, float]]:
        tx = float(np.clip(tx, self.hw[m], self.cw - self.hw[m]))
        ty = float(np.clip(ty, self.hh[m], self.ch - self.hh[m]))
        if not self._overlaps(m, tx, ty):
            return tx, ty
        step = max(self.sizes[m, 0], self.sizes[m, 1]) * 0.25
        for r in range(1, max_rings + 1):
            best: Optional[Tuple[float, float]] = None
            best_d = float("inf")
            for dxi in range(-r, r + 1):
                for dyi in range(-r, r + 1):
                    if abs(dxi) != r and abs(dyi) != r:
                        continue
                    cx = float(np.clip(tx + dxi * step, self.hw[m], self.cw - self.hw[m]))
                    cy = float(np.clip(ty + dyi * step, self.hh[m], self.ch - self.hh[m]))
                    if not self._overlaps(m, cx, cy):
                        d = (cx - tx) ** 2 + (cy - ty) ** 2
                        if d < best_d:
                            best_d = d
                            best = (cx, cy)
            if best is not None:
                return best
        return None

    def _move_macro(self, m: int, nx: float, ny: float) -> float:
        old_x = float(self.pos[m, 0])
        old_y = float(self.pos[m, 1])
        # Remove from grid
        for r, c, a in self._cells_for(m, old_x, old_y):
            self.grid[r, c] -= a
        for r, c, a in self._cells_for(m, nx, ny):
            self.grid[r, c] += a
        # Update pin positions for nets touched by this macro
        for nid, k in self.m_pins[m]:
            self.pin_x[nid][k] = nx + self.net_ox[nid][k]
            self.pin_y[nid][k] = ny + self.net_oy[nid][k]
            px = self.pin_x[nid]
            py = self.pin_y[nid]
            self.net_minx[nid] = px.min()
            self.net_maxx[nid] = px.max()
            self.net_miny[nid] = py.min()
            self.net_maxy[nid] = py.max()
        self.pos[m, 0] = nx
        self.pos[m, 1] = ny
        return math.hypot(nx - old_x, ny - old_y)

    def hpwl_total(self) -> float:
        return float((self.net_maxx - self.net_minx + self.net_maxy - self.net_miny).sum())

    def density_pen(self) -> float:
        return float(((self.grid / self.cell_area) ** 2).sum())

    def cd_pass(self) -> Tuple[int, float]:
        """One CD pass: try to move every movable hard macro to its HPWL target."""
        random.shuffle(self.movable_idx)
        n_acc = 0
        max_disp = 0.0
        cur = self.hpwl_total() + self.lam * self.density_pen()
        for m in self.movable_idx:
            tx, ty = self._compute_target(m)
            legal = self._find_legal(m, tx, ty)
            if legal is None:
                continue
            cx, cy = legal
            if abs(cx - self.pos[m, 0]) < 1e-6 and abs(cy - self.pos[m, 1]) < 1e-6:
                continue
            old_x = float(self.pos[m, 0])
            old_y = float(self.pos[m, 1])
            disp = self._move_macro(m, cx, cy)
            new = self.hpwl_total() + self.lam * self.density_pen()
            if new < cur - 1e-9:
                n_acc += 1
                cur = new
                if disp > max_disp:
                    max_disp = disp
            else:
                # revert
                self._move_macro(m, old_x, old_y)
        self.last_max_disp = max_disp
        self.last_n_acc = n_acc
        return n_acc, cur


# ── Animation harness ──────────────────────────────────────────────────────


def make_animation(
    name: str,
    steps: int = 150,
    fps: int = 12,
    output_format: str = "mp4",
    eval_every: int = 5,
) -> Path:
    bench_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if not bench_dir.exists():
        raise SystemExit(f"Benchmark not found: {bench_dir}")
    benchmark, plc = load_benchmark_from_dir(str(bench_dir))
    placer = _CDPlacer(benchmark, plc)

    nh = benchmark.num_hard_macros
    n = benchmark.num_macros

    # Capture initial frame
    frames: List[dict] = []
    initial_proxy = float(
        compute_proxy_cost(
            torch.tensor(placer.pos, dtype=torch.float32), benchmark, plc
        )["proxy_cost"]
    )
    frames.append(
        dict(
            step=0,
            pos=placer.pos.copy(),
            cost=initial_proxy,
            best=initial_proxy,
            max_disp=0.0,
            net_minx=placer.net_minx.copy(),
            net_maxx=placer.net_maxx.copy(),
            net_miny=placer.net_miny.copy(),
            net_maxy=placer.net_maxy.copy(),
        )
    )

    best = initial_proxy
    print(f"[anim] {name}: initial proxy={initial_proxy:.4f}")
    t0 = time.time()
    for step in range(1, steps + 1):
        n_acc, surrogate = placer.cd_pass()
        # Periodically compute true proxy (slow)
        if step % eval_every == 0 or step == steps:
            cur_proxy = float(
                compute_proxy_cost(
                    torch.tensor(placer.pos, dtype=torch.float32), benchmark, plc
                )["proxy_cost"]
            )
        else:
            cur_proxy = frames[-1]["cost"]  # carry forward
        if cur_proxy < best:
            best = cur_proxy
        frames.append(
            dict(
                step=step,
                pos=placer.pos.copy(),
                cost=cur_proxy,
                best=best,
                max_disp=placer.last_max_disp,
                net_minx=placer.net_minx.copy(),
                net_maxx=placer.net_maxx.copy(),
                net_miny=placer.net_miny.copy(),
                net_maxy=placer.net_maxy.copy(),
            )
        )
        if step % 5 == 0 or step <= 3:
            print(
                f"[anim] step {step:3d}/{steps} acc={n_acc:3d} max_disp={placer.last_max_disp:.2f} "
                f"surrogate={surrogate:.0f} proxy={cur_proxy:.4f} t={time.time()-t0:.1f}s"
            )
        if n_acc == 0:
            print(f"[anim] CD converged at step {step} (no accepted moves)")
            break

    # ── Render animation ─────────────────────────────────────────────
    fig, (ax_pl, ax_cv) = plt.subplots(1, 2, figsize=(14, 6))

    # Static canvas border + axes
    ax_pl.set_xlim(0, benchmark.canvas_width)
    ax_pl.set_ylim(0, benchmark.canvas_height)
    ax_pl.set_aspect("equal")
    ax_pl.set_xlabel("X (μm)")
    ax_pl.set_ylabel("Y (μm)")
    ax_pl.set_title(f"{name} — CD + density placer")
    ax_pl.add_patch(
        Rectangle(
            (0, 0),
            benchmark.canvas_width,
            benchmark.canvas_height,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
    )

    # Initial macros
    rects: List[Rectangle] = []
    for i in range(nh):
        x = frames[0]["pos"][i, 0]
        y = frames[0]["pos"][i, 1]
        w, h = benchmark.macro_sizes[i].tolist()
        is_fixed = bool(benchmark.macro_fixed[i])
        color = "lightcoral" if is_fixed else "steelblue"
        r = Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            facecolor=color,
            alpha=0.55,
            edgecolor="navy" if not is_fixed else "darkred",
            linewidth=0.6,
        )
        rects.append(r)
        ax_pl.add_patch(r)

    # Net spider lines (one segment per (net midpoint -> pin endpoint))
    net_lines = LineCollection([], colors="grey", linewidths=0.4, alpha=0.25, zorder=1)
    ax_pl.add_collection(net_lines)

    # Info box
    info_text = ax_pl.text(
        0.02,
        0.98,
        "",
        transform=ax_pl.transAxes,
        verticalalignment="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    # ── Right panel: cost convergence ────────────────────────────────
    ax_cv.set_xlim(0, len(frames) - 1)
    ymin = min(f["cost"] for f in frames)
    ymax = max(f["cost"] for f in frames)
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
    ax_cv.set_ylim(ymin - pad, ymax + pad)
    ax_cv.set_xlabel("CD pass")
    ax_cv.set_ylabel("Proxy cost")
    ax_cv.set_title("Cost convergence")
    ax_cv.grid(True, alpha=0.3)

    (line_cur,) = ax_cv.plot([], [], color="steelblue", linewidth=1.5, label="current")
    (line_best,) = ax_cv.plot([], [], color="darkgreen", linewidth=1.2, linestyle="--", label="best")
    ax_cv.legend(loc="upper right", fontsize=9)

    # Per-frame net segment cache (compute lazily)
    def build_segments(frame: dict) -> list:
        segs: list = []
        # Sample a subset of nets to keep frames small
        sample = min(800, placer.n_nets)
        idxs = np.linspace(0, placer.n_nets - 1, sample, dtype=int)
        for nid in idxs:
            owners = placer.net_owners[int(nid)]
            isp = placer.net_is_port[int(nid)]
            ox = placer.net_ox[int(nid)]
            oy = placer.net_oy[int(nid)]
            if len(owners) < 2:
                continue
            xs = []
            ys = []
            for k in range(len(owners)):
                if isp[k]:
                    xs.append(ox[k])
                    ys.append(oy[k])
                else:
                    o = int(owners[k])
                    xs.append(frame["pos"][o, 0] + ox[k])
                    ys.append(frame["pos"][o, 1] + oy[k])
            if len(xs) < 2:
                continue
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            for x_i, y_i in zip(xs, ys):
                segs.append([(mx, my), (x_i, y_i)])
        return segs

    def update(frame_idx: int):
        f = frames[frame_idx]
        for i, r in enumerate(rects):
            x = f["pos"][i, 0]
            y = f["pos"][i, 1]
            w, h = benchmark.macro_sizes[i].tolist()
            r.set_xy((x - w / 2, y - h / 2))
        net_lines.set_segments(build_segments(f))
        info_text.set_text(
            f"Step: {f['step']}/{steps}\n"
            f"Max disp: {f['max_disp']:.2f}\n"
            f"Cost:  {f['cost']:.4f}\n"
            f"Best:  {f['best']:.4f}"
        )
        xs = [g["step"] for g in frames[: frame_idx + 1]]
        ys = [g["cost"] for g in frames[: frame_idx + 1]]
        bs = [g["best"] for g in frames[: frame_idx + 1]]
        line_cur.set_data(xs, ys)
        line_best.set_data(xs, bs)
        return rects + [net_lines, info_text, line_cur, line_best]

    print(f"[anim] rendering {len(frames)} frames at {fps} fps...")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000.0 / fps,
        blit=False,
    )

    out_dir = Path("vis")
    out_dir.mkdir(exist_ok=True)
    if output_format == "gif":
        out_path = out_dir / f"{name}_anim.gif"
        ani.save(str(out_path), writer="pillow", fps=fps, dpi=110)
    else:
        out_path = out_dir / f"{name}_anim.mp4"
        try:
            ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=110)
        except (RuntimeError, FileNotFoundError):
            print("[anim] ffmpeg not available — falling back to gif")
            out_path = out_dir / f"{name}_anim.gif"
            ani.save(str(out_path), writer="pillow", fps=fps, dpi=110)
    plt.close(fig)
    print(f"[anim] saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", "-b", default="ibm01")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--gif", action="store_true", help="Force GIF output (default mp4 with fallback)")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Compute true proxy every N CD passes (cheap surrogate is used otherwise)",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    fmt = "gif" if args.gif else "mp4"
    make_animation(
        args.benchmark,
        steps=args.steps,
        fps=args.fps,
        output_format=fmt,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
