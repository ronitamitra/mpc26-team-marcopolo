import torch

def lse_wirelength(positions: torch.Tensor, nets: list, gamma: float = 1.0) -> torch.Tensor:
    """
    Computes the Log-Sum-Exp (LSE) approximation of Half-Perimeter Wirelength (HPWL).
    
    Args:
        positions: [N, 2] tensor of macro centers
        nets: List of lists, where each inner list contains macro indices for a net
        gamma: Smoothing parameter. Smaller gamma approaches exact HPWL but gradients 
               become sharper. Larger gamma provides smoother gradients but looser approx.
               
    Returns:
        Scalar tensor containing the total LSE wirelength.
    """
    total_wl = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
    
    # This is a naive unvectorized loop for skeletal purposes. 
    # For performance, this MUST be vectorized using sparse tensors or gather operations.
    for net in nets:
        if len(net) <= 1:
            continue
            
        net_pos = positions[net] # [num_pins_in_net, 2]
        
        # LSE Formulation for max(x) and min(x)
        max_x = gamma * torch.logsumexp(net_pos[:, 0] / gamma, dim=0)
        min_x = -gamma * torch.logsumexp(-net_pos[:, 0] / gamma, dim=0)
        
        max_y = gamma * torch.logsumexp(net_pos[:, 1] / gamma, dim=0)
        min_y = -gamma * torch.logsumexp(-net_pos[:, 1] / gamma, dim=0)
        
        total_wl = total_wl + (max_x - min_x) + (max_y - min_y)
        
    return total_wl
