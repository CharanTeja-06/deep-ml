def compute_efficiency(n_experts: int, k_active: int, d_in: int, d_out: int) -> float:
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts per token (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    # Guard against invalid inputs or division by zero
    if n_experts <= 0 or k_active <= 0 or k_active > n_experts:
        return 0.0

    # Linear layer forward pass FLOPs = 2 * d_in * d_out (multiply-accumulate)
    expert_flops = 2 * d_in * d_out
    
    # Dense baseline evaluates all experts for the equivalent parameter capacity
    dense_flops = n_experts * expert_flops
    
    # MoE baseline only evaluates the top-k active experts (ignoring trivial routing overhead)
    moe_flops = k_active * expert_flops
    
    # Calculate percentage reduction in computation
    flop_savings = ((dense_flops - moe_flops) / dense_flops) * 100
    
    return flop_savings
