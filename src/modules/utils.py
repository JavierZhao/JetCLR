import torch
from copy import deepcopy
def generate_mask(x):
    """
    Generate a mask, real particle = 1, padded = 0
    Input:
        x: torch.Tensor of shape (batch_size, 6, num_particles)
    Output:
        mask: torch.Tensor of shape (batch_size, 1, num_particles)
    """
    mask = x.clone()
    non_zero_mask = mask.any(dim=1, keepdim=True)  
    print(non_zero_mask)

    mask = non_zero_mask.int()

    return mask
    
def calculate_cartesian_components(input_tensor):
    # Input tensor shape: (batch_size, 6, 128)
    # Extract components
    eta = input_tensor[:, 0, :]  # part_eta
    phi = input_tensor[:, 1, :]  # part_phi
    log_pt = input_tensor[:, 2, :]  # part_pt_log
    log_e = input_tensor[:, 3, :]  # part_e_log

    # Calculate pT and E from their logarithmic forms
    pT = torch.exp(log_pt)
    E = torch.exp(log_e)

    # Calculate Cartesian components
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(eta)

    # Stack the components to form the output tensor
    output_tensor = torch.stack([px, py, pz, E], dim=1)  # Reordering to (batch_size, 4, 128)

    return output_tensor