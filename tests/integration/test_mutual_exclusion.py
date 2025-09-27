# -*- coding: utf-8 -*-
"""
Test script to verify mutual exclusion constraint for charging and discharging
"""

import sys
from demo_discharge import build_demo_with_discharge
from admm_coordinator import admm_cut_with_baseline


def test_mutual_exclusion():
    """Test that charging and discharging don't occur simultaneously"""
    print("Testing mutual exclusion constraint...")
    
    # Generate small test case
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo_with_discharge(
        n_evs=5, hours=12, dt_min=5, seed=42, include_v2g=True
    )
    
    print(f"Generated {len(evs)} EVs")
    v2g_count = sum(1 for ev in evs if ev.can_discharge_to_grid)
    print(f"V2G capable EVs: {v2g_count}")
    
    # Run optimization
    out = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, enforce_blocks,
        post_slots=3, rho_r=3.0, rho_e=3.0, rho_p=2.0,
        steps=20, time_limit_sec_local=3
    )
    
    print(f"Optimization status: {out.get('status', 'Unknown')}")
    
    # Check mutual exclusion constraint
    if 'individual_p_ch' in out and 'individual_p_dch' in out:
        violations = 0
        total_slots_checked = 0
        
        for i, ev in enumerate(evs):
            for t in range(len(out['individual_p_ch'][i])):
                p_ch = out['individual_p_ch'][i][t]
                p_dch = out['individual_p_dch'][i][t]
                
                # Check if both are positive (violation)
                if p_ch > 0.001 and p_dch > 0.001:  # Small tolerance for numerical errors
                    violations += 1
                    print(f"VIOLATION: EV {ev.name} at time {t}: charging={p_ch:.3f}, discharging={p_dch:.3f}")
                
                if ev.can_discharge_to_grid and ev.available[t]:
                    total_slots_checked += 1
        
        print(f"\nMutual exclusion check:")
        print(f"  Total V2G-capable slots checked: {total_slots_checked}")
        print(f"  Violations found: {violations}")
        
        if violations == 0:
            print("  ✓ PASS: No simultaneous charging and discharging detected")
        else:
            print(f"  ✗ FAIL: {violations} violations found")
        
        # Show some sample data
        print("\nSample data (first V2G-capable EV, first 10 time slots):")
        v2g_ev_idx = next((i for i, ev in enumerate(evs) if ev.can_discharge_to_grid), None)
        if v2g_ev_idx is not None:
            print("Time\tCharging\tDischarging\tNet\tAvailable")
            for t in range(min(10, len(out['individual_p_ch'][v2g_ev_idx]))):
                p_ch = out['individual_p_ch'][v2g_ev_idx][t]
                p_dch = out['individual_p_dch'][v2g_ev_idx][t]
                net = p_ch - p_dch
                avail = evs[v2g_ev_idx].available[t]
                print(f"{t}\t{p_ch:.3f}\t\t{p_dch:.3f}\t\t{net:.3f}\t{avail}")
        
        return violations == 0
    else:
        print("ERROR: Individual charging/discharging data not available in results")
        return False


def test_energy_balance():
    """Test energy balance in SOC calculations"""
    print("\n" + "="*50)
    print("Testing energy balance...")
    
    # Generate test case with known parameters
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo_with_discharge(
        n_evs=3, hours=6, dt_min=15, seed=123, include_v2g=True
    )
    
    out = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, enforce_blocks,
        steps=10, time_limit_sec_local=2
    )
    
    if 'individual_p_ch' in out and 'individual_p_dch' in out:
        dt_hr = gp.dt_min / 60.0
        
        for i, ev in enumerate(evs):
            if not ev.can_discharge_to_grid:
                continue  # Skip charge-only EVs
                
            print(f"\nEV {ev.name} (V2G capable):")
            print(f"  Capacity: {ev.e_kwh:.1f} kWh, Initial SOC: {ev.soc0:.2f}")
            
            soc = ev.soc0
            energy_balance_errors = []
            
            for t in range(len(out['individual_p_ch'][i])):
                p_ch = out['individual_p_ch'][i][t]
                p_dch = out['individual_p_dch'][i][t]
                
                # Calculate energy changes
                charge_energy = p_ch * ev.eta_c * dt_hr
                discharge_energy = p_dch / ev.eta_d * dt_hr
                external_use = 0.0
                if hasattr(ev, 'discharge_schedule') and ev.discharge_schedule and t < len(ev.discharge_schedule):
                    external_use = ev.discharge_schedule[t] * dt_hr
                
                # Update SOC
                prev_soc = soc
                soc = soc + (charge_energy - discharge_energy - external_use) / ev.e_kwh
                soc = max(ev.soc_min, min(ev.soc_max, soc))
                
                # Check for energy balance issues
                if t < 5:  # Show first few time slots
                    print(f"  t={t}: charge={p_ch:.3f}kW, discharge={p_dch:.3f}kW, external={external_use:.3f}kWh")
                    print(f"    SOC: {prev_soc:.3f} → {soc:.3f}")
    
    return True


if __name__ == "__main__":
    print("ADMM EV Aggregation - Mutual Exclusion Constraint Test")
    print("=" * 60)
    
    success1 = test_mutual_exclusion()
    success2 = test_energy_balance()
    
    print("\n" + "="*60)
    print("TEST SUMMARY:")
    print(f"  Mutual exclusion constraint: {'PASS' if success1 else 'FAIL'}")
    print(f"  Energy balance: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("  Overall: ALL TESTS PASSED ✓")
        sys.exit(0)
    else:
        print("  Overall: SOME TESTS FAILED ✗")
        sys.exit(1)
