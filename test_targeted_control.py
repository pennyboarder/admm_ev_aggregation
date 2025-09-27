# -*- coding: utf-8 -*-
"""
Test targeted baseline control with coordinator-EV communication
"""

from demo import build_demo
from admm_coordinator import admm_cut_with_baseline

def test_targeted_baseline_control():
    """Test the new targeted baseline control feature"""
    print("=" * 60)
    print("TESTING TARGETED BASELINE CONTROL")
    print("=" * 60)
    
    # Generate small demo for easier observation
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo(n_evs=5, hours=3, dt_min=5, seed=42)
    
    print(f"Number of EVs: {len(evs)}")
    print(f"Time slots: {len(P_base)}")
    print(f"Blocks: {len(R_bid)}")
    print(f"Baseline requirement: {P_base[0]:.1f}kW")
    
    # Check total EV capacity
    total_max_power = sum(ev.p_charge_max for ev in evs)
    print(f"Total max power: {total_max_power:.1f}kW")
    
    # Force all EVs to be available for testing
    for ev in evs:
        ev.available = [True] * len(ev.available)
    
    # Test with targeted bidding strategy (bid in first block only)
    targeted_bidding = [1, 0, 0, 0, 0, 0]  # Only first block
    
    print(f"\n🎯 Testing targeted baseline control")
    print(f"Winning blocks: {[i for i, b in enumerate(targeted_bidding) if b == 1]}")
    print(f"Baseline target: {P_base[0]:.1f}kW")
    
    # Run ADMM with targeted baseline control
    from admm_coordinator import admm_cut_with_baseline_oop
    out = admm_cut_with_baseline_oop(
        evs, gp, mi, R_bid, P_base, targeted_bidding,
        post_slots=6, rho_r=6.0, rho_e=4.0, rho_p=3.0,
        steps=10, time_limit_sec_local=5, over_relax=1.0,
        num_threads=1,  # Single thread for clear output
        enforce_baseline_for_reserve=False  # Use targeted control instead
    )
    
    print("\n--- Results ---")
    agg_p = out["sum_p_ch"]
    mask = out["baseline_mask_t"]
    
    print(f"Reserve capacity: {[round(x,1) for x in out['sum_r_cut']]}")
    print(f"Baseline mask (first 18): {mask[:18]}")
    print(f"Aggregate power (first 18): {[round(p,1) for p in agg_p[:18]]}")
    print(f"Target baseline (first 18): {[round(p,1) for p in P_base[:18]]}")
    
    # Analyze baseline tracking in first block (slots 0-5)
    print(f"\n🎯 Baseline tracking analysis for first block (slots 0-5):")
    block_0_slots = list(range(6))  # First 6 slots = first block
    total_error = 0
    for t in block_0_slots:
        if t < len(agg_p) and t < len(P_base):
            actual = agg_p[t]
            target = P_base[t]
            error = abs(actual - target)
            error_pct = (error / target * 100) if target > 0 else 0
            total_error += error
            status = "✅" if error < 2.0 else "❌"
            print(f"  {status} Slot {t}: target={target:.1f}kW, actual={actual:.1f}kW, error={error:.1f}kW ({error_pct:.1f}%)")
    
    avg_error = total_error / len(block_0_slots)
    print(f"\n📊 Average error in winning block: {avg_error:.1f}kW")
    
    if avg_error < 3.0:
        print("🎉 SUCCESS: Targeted baseline control is working!")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Baseline tracking still has significant errors")
    
    return out

if __name__ == "__main__":
    test_targeted_baseline_control()
