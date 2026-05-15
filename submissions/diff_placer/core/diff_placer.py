import torch
import torch.nn as nn
from macro_place.benchmark import Benchmark

class DiffPlacerModule(nn.Module):
    """
    A PyTorch Module holding the learnable macro positions.
    """
    def __init__(self, benchmark: Benchmark):
        super().__init__()
        
        # Clone initial positions
        init_pos = benchmark.macro_positions.clone()
        
        # In a real implementation, you might want to separate hard and soft macros 
        # into different parameter tensors so they can be frozen/updated independently.
        self.positions = nn.Parameter(init_pos)
        
        # Store masks to easily identify macro types during CD loop
        self.register_buffer("hard_macro_mask", benchmark.get_hard_macro_mask())
        self.register_buffer("movable_mask", benchmark.get_movable_mask())
        
        # Get canvas bounds for legalization/clipping
        self.canvas_w = benchmark.canvas_width
        self.canvas_h = benchmark.canvas_height

    def forward(self) -> torch.Tensor:
        """
        Forward pass simply returns the current placement positions, 
        potentially clamped to the canvas boundaries.
        """
        # Ensure positions don't drift completely out of bounds
        return torch.clamp(
            self.positions, 
            min=0.0, 
            max=max(self.canvas_w, self.canvas_h) # Simplified clamping
        )
