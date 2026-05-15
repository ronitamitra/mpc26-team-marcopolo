"""
Differentiable PyTorch Placer Entry Point
"""

import torch
from macro_place.benchmark import Benchmark
from .core.diff_placer import DiffPlacerModule
from .optimizer.cd_loop import optimize_placement

class DifferentiablePlacer:
    """
    A PyTorch-based differentiable analytical placer using Coordinate Descent.
    """

    def __init__(self, learning_rate: float = 0.1, max_iters: int = 100):
        self.learning_rate = learning_rate
        self.max_iters = max_iters

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Main entry point for the evaluation framework.
        """
        # Determine the device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        print(f"[DiffPlacer] Running on device: {device}")

        # Initialize the placement module with the initial positions
        placer_module = DiffPlacerModule(benchmark).to(device)

        # Run the Coordinate Descent optimization loop
        final_placement = optimize_placement(
            placer_module=placer_module,
            benchmark=benchmark,
            device=device,
            learning_rate=self.learning_rate,
            max_iters=self.max_iters
        )

        return final_placement.cpu()
