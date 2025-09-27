# -*- coding: utf-8 -*-
"""
Test script for EV discharge functionality
"""

from demo_discharge import build_demo_with_discharge
from admm_coordinator import admm_cut_with_baseline
from visualization import (
    plot_ev_charging_schedule,
    plot_soc_evolution,
    plot_ev_discharge_patterns,
    save_all_plots
)
from csv_export import export_all_csv_data


def test_discharge_scenario():
    """Test EV discharge scenario with external use patterns"""
    print("="*60)
    print("EV DISCHARGE SCENARIO TEST")
    print("="*60)
    
    # Generate demo with discharge patterns
    print("Generating demo with discharge patterns...")
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo_with_discharge(
        n_evs=6, 
        hours=12, 
        dt_min=5, 
        seed=42,
        include_v2g=True,  # Enable V2G for some EVs
        discharge_intensity=0.6  # Moderate discharge activity
    )
    
    print(f"Created {len(evs)} EVs:")
    for i, ev in enumerate(evs):
        has_discharge = sum(ev.discharge_schedule) > 0 if ev.discharge_schedule else False
        total_discharge = sum(ev.discharge_schedule) if ev.discharge_schedule else 0
        print(f"  {ev.name}: SOC={ev.soc0:.2f}, Capacity={ev.e_kwh:.1f}kWh, "
              f"V2G={ev.can_discharge_to_grid}, ExtUse={total_discharge:.1f}kWh")
    
    # Run optimization
    print("\nRunning ADMM optimization with discharge patterns...")
    results = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, enforce_blocks,
        post_slots=6, rho_r=8.0, rho_e=6.0, rho_p=4.0,
        steps=25, time_limit_sec_local=8, over_relax=1.15
    )
    
    # Results summary
    print(f"\nOptimization completed:")
    print(f"  ADMM iterations: {len(results['history'])}")
    print(f"  Final residual: {results['history'][-1]['r_norm']:.6f}")
    print(f"  Total reserve capacity: {sum(results['sum_r_cut']):.1f} kW")
    print(f"  Total recovery energy: {sum(results['sum_e_rec']):.1f} kWh")
    print(f"  Enforced blocks: {sum(enforce_blocks)} out of {len(enforce_blocks)}")
    
    # Create output directory
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/discharge_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations in: {output_dir}")
    
    # Enhanced visualizations
    plot_ev_charging_schedule(
        evs, gp, results, 
        ev_indices=[0,1,2,3],
        save_path=f"{output_dir}/charging_schedule.png"
    )
    
    plot_soc_evolution(
        evs, gp, results,
        ev_indices=[0,1,2,3], 
        save_path=f"{output_dir}/soc_evolution.png"
    )
    
    # New: discharge patterns
    plot_ev_discharge_patterns(
        evs, gp, results,
        ev_indices=[0,1,2,3],
        save_path=f"{output_dir}/discharge_patterns.png"
    )
    
    # Export CSV data
    print("Exporting detailed CSV data...")
    csv_dir = export_all_csv_data(
        evs, gp, mi, results, P_base, enforce_blocks, 
        output_dir=output_dir,
        prefix="discharge_test"
    )
    
    print(f"\nTest completed! All results saved to: {output_dir}")
    return output_dir


def compare_scenarios():
    """Compare scenarios with and without discharge"""
    print("\n" + "="*60)
    print("SCENARIO COMPARISON: WITH vs WITHOUT DISCHARGE")
    print("="*60)
    
    scenarios = [
        ("No Discharge", {"discharge_intensity": 0.0, "include_v2g": False}),
        ("Light Discharge", {"discharge_intensity": 0.3, "include_v2g": False}),
        ("Heavy Discharge + V2G", {"discharge_intensity": 0.7, "include_v2g": True}),
    ]
    
    results_comparison = []
    
    for scenario_name, params in scenarios:
        print(f"\nTesting scenario: {scenario_name}")
        
        evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo_with_discharge(
            n_evs=4, hours=8, dt_min=5, seed=123, **params
        )
        
        results = admm_cut_with_baseline(
            evs, gp, mi, R_bid, P_base, enforce_blocks,
            steps=20, rho_r=6.0, rho_e=4.0, rho_p=3.0
        )
        
        # Calculate metrics
        total_charging = sum(results['sum_p_ch']) * gp.dt_min / 60  # kWh
        total_reserve = sum(results['sum_r_cut'])
        final_residual = results['history'][-1]['r_norm'] if results['history'] else 0
        
        # Calculate total discharge
        total_ext_discharge = 0
        for ev in evs:
            if ev.discharge_schedule:
                total_ext_discharge += sum(ev.discharge_schedule)
        
        results_comparison.append({
            'scenario': scenario_name,
            'total_charging_kwh': total_charging,
            'total_reserve_kw': total_reserve,
            'total_ext_discharge_kwh': total_ext_discharge,
            'final_residual': final_residual,
            'iterations': len(results['history'])
        })
    
    # Print comparison table
    print(f"\n{'Scenario':<20} {'Charging':<10} {'Reserve':<10} {'ExtUse':<10} {'Residual':<12} {'Iters':<6}")
    print("-" * 75)
    for r in results_comparison:
        print(f"{r['scenario']:<20} {r['total_charging_kwh']:<10.1f} "
              f"{r['total_reserve_kw']:<10.1f} {r['total_ext_discharge_kwh']:<10.1f} "
              f"{r['final_residual']:<12.6f} {r['iterations']:<6}")


if __name__ == "__main__":
    # Run discharge scenario test
    output_dir = test_discharge_scenario()
    
    # Compare different scenarios
    compare_scenarios()
    
    print(f"\nAll tests completed! Main results in: {output_dir}")
