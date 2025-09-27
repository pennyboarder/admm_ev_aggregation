# -*- coding: utf-8 -*-
"""
Simple test to debug baseline tracking issue
"""

from demo import build_demo
from admm_coordinator import admm_cut_with_baseline
from bidding_optimizer import solve_ev_with_bidding_optimization

def create_block_mapping(B, T, slots_per_block):
    """Create mapping from blocks to time slots"""
    blocks = []
    for b in range(B):
        start_slot = b * slots_per_block
        end_slot = min(start_slot + slots_per_block, T)
        block_slots = list(range(start_slot, end_slot))
        blocks.append(block_slots)
    return blocks

def debug_baseline_tracking():
    """Debug baseline tracking with a simple case"""
    print("=" * 60)
    print("DEBUGGING BASELINE TRACKING")
    print("=" * 60)
    
    # Use small demo for easier debugging
    print("Generating small demo data...")
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo(n_evs=3, hours=4, dt_min=5, seed=1)
    
    print(f"Number of EVs: {len(evs)}")
    print(f"Time slots: {len(P_base)}")
    print(f"Blocks: {len(R_bid)}")
    print(f"Original enforce_blocks: {enforce_blocks}")
    print(f"P_base (first 12): {P_base[:12]}")
    
    # Test with fixed bidding strategy (bid in first 2 blocks only)
    print("\n--- Testing with fixed bidding strategy ---")
    fixed_bidding = [1, 1, 0, 0, 0, 0, 0, 0]  # Only bid in first 2 blocks
    
    print(f"Fixed bidding strategy: {fixed_bidding}")
    
    # Run ADMM with baseline enforcement
    print("\nRunning ADMM with baseline enforcement...")
    out = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, fixed_bidding,
        post_slots=6, rho_r=6.0, rho_e=4.0, rho_p=3.0,
        steps=10, time_limit_sec_local=3, over_relax=1.0,
        num_threads=1,  # Single thread for debugging
        enforce_baseline_for_reserve=True
    )
    
    print("\n--- Results ---")
    print(f"ADMM convergence: {len(out['history'])} iterations")
    if out["history"]:
        print(f"Final residual: {out['history'][-1]['r_norm']:.6f}")
    
    print(f"Sum r_cut [kW]: {[round(x,2) for x in out['sum_r_cut']]}")
    print(f"Sum e_rec [kWh]: {[round(x,2) for x in out['sum_e_rec']]}")
    
    # Check baseline tracking in detail
    agg_p = out["sum_p_ch"]
    mask = out["baseline_mask_t"]
    
    print(f"\nBaseline mask (first 24): {mask[:24]}")
    print(f"Aggregate power (first 24): {[round(p,1) for p in agg_p[:24]]}")
    print(f"Target baseline (first 24): {[round(p,1) for p in P_base[:24]]}")
    
    # Calculate errors for enforced slots
    enforced_slots = [i for i, m in enumerate(mask) if m == 1][:12]
    if enforced_slots:
        print(f"\nEnforced slots: {enforced_slots}")
        for t in enforced_slots:
            if t < len(agg_p) and t < len(P_base):
                actual = agg_p[t]
                target = P_base[t]
                error = actual - target
                error_pct = (error / target * 100) if target > 0 else 0
                print(f"  Slot {t}: target={target:.1f}, actual={actual:.1f}, error={error:.1f} ({error_pct:.1f}%)")

if __name__ == "__main__":
    debug_baseline_tracking()
