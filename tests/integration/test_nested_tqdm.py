# -*- coding: utf-8 -*-
"""
Test script for nested tqdm progress bars with large dataset
"""

from demo_discharge import build_demo_with_discharge
from admm_coordinator import admm_cut_with_baseline


def test_large_dataset():
    """Test ADMM with large number of EVs to see nested progress bars"""
    print("Testing nested progress bars with large dataset...")
    
    # Generate larger dataset to trigger nested progress bars
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo_with_discharge(
        n_evs=20, hours=4, dt_min=10, seed=456, include_v2g=True
    )
    
    print(f"Generated {len(evs)} EVs for nested progress test")
    
    # Run with fewer iterations but show nested progress
    out = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, enforce_blocks,
        post_slots=2, rho_r=2.0, rho_e=2.0, rho_p=2.0,
        steps=5, time_limit_sec_local=1, over_relax=1.0
    )
    
    print(f"Completed optimization")
    return out


if __name__ == "__main__":
    test_large_dataset()
