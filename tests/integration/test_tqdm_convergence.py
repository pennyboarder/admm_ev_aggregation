# -*- coding: utf-8 -*-
"""
Test script for tqdm progress visualization with early convergence
"""

from demo_discharge import build_demo_with_discharge
from admm_coordinator import admm_cut_with_baseline


def test_early_convergence():
    """Test ADMM with tight convergence criteria to see early stopping"""
    print("Testing early convergence with tqdm...")
    
    # Generate small test case that should converge quickly
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo_with_discharge(
        n_evs=5, hours=6, dt_min=15, seed=123, include_v2g=True
    )
    
    print(f"Generated {len(evs)} EVs for convergence test")
    
    # Run with more lenient parameters for faster convergence
    out = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, enforce_blocks,
        post_slots=2, rho_r=1.0, rho_e=1.0, rho_p=1.0,
        steps=50, time_limit_sec_local=2, over_relax=1.0
    )
    
    if out["history"]:
        final_iter = len(out["history"])
        final_residual = out["history"][-1]["r_norm"]
        print(f"\nConverged after {final_iter} iterations")
        print(f"Final residual: {final_residual:.6f}")
        
        if final_iter < 50:
            print("✓ Early convergence detected!")
        else:
            print("Completed all iterations")
    
    return out


if __name__ == "__main__":
    test_early_convergence()
