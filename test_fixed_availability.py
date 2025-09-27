# Test with better EV availability
from demo import build_demo
from admm_coordinator import admm_cut_with_baseline

def test_with_better_availability():
    print("=" * 60)
    print("TESTING WITH IMPROVED EV AVAILABILITY")
    print("=" * 60)
    
    # Generate demo data
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo(n_evs=10, hours=4, dt_min=5, seed=42)
    
    # Manually fix EV availability to ensure sufficient power
    print("Original EV availability:")
    total_available_power = []
    for t in range(12):
        power_t = sum(ev.p_charge_max for ev in evs if ev.available[t])
        total_available_power.append(power_t)
    
    print(f"Available power per slot: {[round(p,1) for p in total_available_power[:12]]}")
    print(f"Baseline requirement: {P_base[0]:.1f}kW")
    
    # Force all EVs to be available at all times for testing
    print("\nForcing all EVs to be available...")
    for ev in evs:
        ev.available = [True] * len(ev.available)
    
    # Recalculate available power
    total_max_power = sum(ev.p_charge_max for ev in evs)
    print(f"Total maximum power after fix: {total_max_power:.1f}kW")
    print(f"Can meet baseline? {'Yes' if total_max_power >= P_base[0] else 'No'}")
    
    # Test with fixed bidding strategy
    fixed_bidding = [1, 1] + [0] * (len(R_bid) - 2)  # Bid in first 2 blocks
    
    print(f"\nTesting with bidding strategy: {fixed_bidding}")
    
    # Run ADMM with baseline enforcement
    out = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, fixed_bidding,
        post_slots=6, rho_r=6.0, rho_e=4.0, rho_p=3.0,
        steps=15, time_limit_sec_local=5, over_relax=1.0,
        num_threads=1,
        enforce_baseline_for_reserve=True
    )
    
    print("\n--- Results after availability fix ---")
    agg_p = out["sum_p_ch"]
    mask = out["baseline_mask_t"]
    
    print(f"Sum r_cut [kW]: {[round(x,1) for x in out['sum_r_cut']]}")
    print(f"Baseline mask (first 12): {mask[:12]}")
    print(f"Aggregate power (first 12): {[round(p,1) for p in agg_p[:12]]}")
    print(f"Target baseline (first 12): {[round(p,1) for p in P_base[:12]]}")
    
    # Check baseline adherence
    enforced_slots = [i for i, m in enumerate(mask) if m == 1][:12]
    if enforced_slots:
        print(f"\nBaseline tracking results:")
        total_error = 0
        for t in enforced_slots:
            if t < len(agg_p) and t < len(P_base):
                actual = agg_p[t]
                target = P_base[t]
                error = abs(actual - target)
                total_error += error
                print(f"  Slot {t}: target={target:.1f}, actual={actual:.1f}, error={error:.1f}")
        
        avg_error = total_error / len(enforced_slots) if enforced_slots else 0
        print(f"\nAverage baseline error: {avg_error:.2f}kW")
        if avg_error < 0.1:
            print("✅ Baseline tracking SUCCESSFUL!")
        else:
            print("❌ Baseline tracking still has issues")

if __name__ == "__main__":
    test_with_better_availability()
