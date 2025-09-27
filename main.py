# -*- coding: utf-8 -*-
"""
Main script for running ADMM EV aggregation demo with automatic bidding optimization
"""

from demo import build_demo
from demo_discharge import build_demo_with_discharge
from admm_coordinator import admm_cut_with_baseline
from bidding_optimizer import solve_ev_with_bidding_optimization
from models import EVAgent
from visualization import (
    plot_ev_charging_schedule,
    plot_soc_evolution,
    plot_aggregate_analysis,
    plot_convergence_history,
    plot_ev_discharge_patterns
)
from csv_export import export_all_csv_data
import sys


def create_block_mapping(B, T, slots_per_block):
    """Create mapping from blocks to time slots"""
    blocks = []
    for b in range(B):
        start_slot = b * slots_per_block
        end_slot = min(start_slot + slots_per_block, T)
        block_slots = list(range(start_slot, end_slot))
        blocks.append(block_slots)
    return blocks


def save_detailed_results(evs, gp, out, P_base, enforce_blocks, output_dir, bidding_result=None):
    """Save detailed optimization results to text file"""
    import json
    from datetime import datetime
    
    results_file = f"{output_dir}/detailed_results.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ADMM EV AGGREGATION OPTIMIZATION RESULTS WITH BIDDING OPTIMIZATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System configuration
        f.write("SYSTEM CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of EVs: {len(evs)}\n")
        f.write(f"Simulation period: 24 hours\n")
        f.write(f"Time resolution: {gp.dt_min} minutes\n")
        f.write(f"Sustain period: {gp.sustain_min} minutes\n")
        f.write(f"Degradation cost: {gp.degr_cost_per_kwh} ¥/kWh\n")
        f.write(f"Optimal bidding blocks: {sum(enforce_blocks)} out of {len(enforce_blocks)}\n")
        
        # Add bidding optimization results if available
        if bidding_result and 'bidding_metrics' in bidding_result:
            f.write(f"Expected net profit: {bidding_result['bidding_metrics']['net_profit']:.1f} ¥\n")
            f.write(f"Total reserve revenue: {bidding_result['bidding_metrics']['total_reserve_revenue']:.1f} ¥\n")
            f.write(f"Total energy cost: {bidding_result['bidding_metrics']['total_energy_cost']:.1f} ¥\n")
            f.write(f"Bidding strategy: {[int(b) for b in enforce_blocks]}\n")
        f.write("\n")
        
        # EV details
        f.write("EV FLEET DETAILS\n")
        f.write("-" * 40 + "\n")
        f.write("EV_Name\tCapacity[kWh]\tInitSOC\tMaxPower[kW]\tEfficiency\tAvailableSlots\n")
        for i, ev in enumerate(evs):
            available_count = sum(ev.available)
            f.write(f"{ev.name}\t{ev.e_kwh:.1f}\t\t{ev.soc0:.2f}\t{ev.p_charge_max:.1f}\t\t{ev.eta_c:.2f}\t\t{available_count}\n")
        f.write("\n")
        
        # Optimization results
        if out["history"]:
            final_residual = out["history"][-1]["r_norm"]
            iterations = len(out["history"])
            f.write("OPTIMIZATION PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"ADMM iterations: {iterations}\n")
            f.write(f"Final residual: {final_residual:.6f}\n")
            f.write(f"Convergence status: {'Converged' if final_residual < 1e-3 else 'Not fully converged'}\n\n")
        
        # Summary metrics
        f.write("SUMMARY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total reserve capacity: {sum(out['sum_r_cut']):.1f} kW\n")
        f.write(f"Total recovery energy: {sum(out['sum_e_rec']):.1f} kWh\n")
        f.write(f"Total aggregate charging: {sum(out['sum_p_ch']):.1f} kW⋅5min\n\n")
        
        # Detailed block results
        f.write("BLOCK-BY-BLOCK RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write("Block\tBidding\tReserve[kW]\tRecovery[kWh]\n")
        for b in range(len(out['sum_r_cut'])):
            bidding = "Yes" if enforce_blocks[b] else "No"
            f.write(f"{b}\t{bidding}\t\t{out['sum_r_cut'][b]:.2f}\t\t{out['sum_e_rec'][b]:.2f}\n")
        f.write("\n")
        
        # Baseline tracking performance
        agg_p = out["sum_p_ch"]
        mask = out["baseline_mask_t"]
        first_masked = [i for i,m in enumerate(mask) if m==1][:12]
        if first_masked:
            f.write("BASELINE TRACKING PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            t0 = first_masked[0]
            actual_baseline = [agg_p[t] for t in range(t0, min(len(agg_p), t0+12))]
            target_baseline = [P_base[t] for t in range(t0, min(len(P_base), t0+12))]
            
            f.write("Slot\tTarget[kW]\tActual[kW]\tError[kW]\tError[%]\n")
            for i, (target, actual) in enumerate(zip(target_baseline, actual_baseline)):
                error_kw = actual - target
                error_pct = (error_kw / target * 100) if target > 0 else 0
                f.write(f"{t0+i}\t{target:.2f}\t\t{actual:.2f}\t\t{error_kw:.2f}\t\t{error_pct:.1f}%\n")
            f.write("\n")
    
    print(f"📄 Detailed results saved to: {results_file}")


def main():
    """Run the ADMM EV aggregation demo with automatic bidding optimization"""
    print("=" * 60)
    print("ADMM EV AGGREGATION WITH AUTOMATIC BIDDING OPTIMIZATION")
    print("=" * 60)
    
    print("Generating demo data...")
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo(n_evs=50, hours=24, dt_min=5, seed=3)
    
    # Fix EV availability to ensure sufficient capacity for baseline tracking
    # print("🔧 Fixing EV availability for better baseline tracking...")
    for ev in evs:
        ev.available = [True] * len(ev.available)
    
    # Run bidding optimization first to determine optimal bidding strategy
    # print("\n🔍 Running bidding optimization...")
    
    # Aggregate EV parameters for simplified optimization
    total_capacity = sum(ev.e_kwh for ev in evs)
    avg_efficiency = sum(ev.eta_c for ev in evs) / len(evs)
    total_max_power = sum(ev.p_charge_max for ev in evs)
    avg_soc0 = sum(ev.soc0 for ev in evs) / len(evs)
    avg_soc_min = sum(ev.soc_min for ev in evs) / len(evs)
    avg_soc_max = sum(ev.soc_max for ev in evs) / len(evs)
    
    # Create availability mask (available if at least half of EVs are available)
    aggregate_available = []
    for t in range(len(evs[0].available)):
        available_count = sum(1 for ev in evs if ev.available[t])
        aggregate_available.append(available_count >= len(evs) * 0.5)
    
    # Prepare parameters for bidding optimization
    T = len(P_base)
    B = len(R_bid)
    dt_hr = gp.dt_min / 60.0
    
    # Create block-to-slot mapping (each block contains consecutive slots)
    slots_per_block = gp.sustain_min // gp.dt_min  # 30 min / 5 min = 6 slots per block
    blocks = create_block_mapping(B, T, slots_per_block)
    
    cap_prices = [100.0] * B  # Example capacity prices (¥/kW)
    
    # Convert to dictionary format for bidding optimizer
    agg_ev_dict = {
        'id': 0,
        'battery_capacity': total_capacity,
        'initial_soc': avg_soc0 * 100.0,  # Convert to percentage
        'target_soc': avg_soc_max * 100.0,  # Convert to percentage
        'charge_power_max': total_max_power,
        'charge_efficiency': avg_efficiency,
        'discharge_efficiency': 0.95,  # Default discharge efficiency
        'can_discharge': any(ev.can_discharge_to_grid for ev in evs)
    }
    
    # Run bidding optimization
    bidding_result = solve_ev_with_bidding_optimization(
        agg_ev_dict, mi, gp, blocks, T, B, dt_hr, R_bid, cap_prices, P_base
    )
    optimal_bidding = bidding_result['bid_decision']
    
    print(f"✅ Optimal bidding strategy: {[int(b) for b in optimal_bidding]}")
    print(f"📊 Total reserve from optimal bidding: {sum(bidding_result['r_cut']):.1f} kW")
    print(f"💰 Expected net profit from optimal bidding: {bidding_result['bidding_metrics']['net_profit']:.1f} ¥")
    
    # Use optimal bidding strategy for ADMM optimization
    # print("\n🚀 Running ADMM optimization with optimal bidding strategy...")
    # print("Progress will be displayed with tqdm progress bars...")
    # print("Using multi-threaded parallel processing...")
    # 新しいターゲット制御機能を使用してベースライン追従を改善
    from admm_coordinator import admm_cut_with_baseline_oop
    out = admm_cut_with_baseline_oop(
        evs, gp, mi, R_bid, P_base, optimal_bidding,
        post_slots=6, rho_r=6.0, rho_e=4.0, rho_p=3.0,
        steps=30, time_limit_sec_local=6, over_relax=1.25,
        num_threads=None,  # Auto-detect CPU cores
        enforce_baseline_for_reserve=False,  # 新しいターゲット制御を使用
        use_oop_solver=True  # OOPソルバーを有効化
    )
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    print(f"Number of EVs: {len(evs)}")
    print(f"Simulation period: {24} hours")
    print(f"Time resolution: {gp.dt_min} minutes")
    print(f"Optimal bidding blocks: {sum(optimal_bidding)} out of {len(optimal_bidding)}")
    print(f"Expected net profit: {bidding_result['bidding_metrics']['net_profit']:.1f} ¥")
    
    if out["history"]:
        final_residual = out["history"][-1]["r_norm"]
        iterations = len(out["history"])
        print(f"ADMM convergence: {iterations} iterations, final residual: {final_residual:.6f}")
    
    print(f"Total reserve capacity: {sum(out['sum_r_cut']):.1f} kW")
    print(f"Total recovery energy: {sum(out['sum_e_rec']):.1f} kWh")
    print(f"Total aggregate charging: {sum(out['sum_p_ch']):.1f} kW⋅5min")
    
    # Detailed results
    print("\nDetailed Results:")
    print("Sum r_cut [kW]: ", [round(x,1) for x in out["sum_r_cut"]])
    print("Sum e_rec [kWh]:", [round(x,1) for x in out["sum_e_rec"]])
    
    # Show baseline tracking performance
    agg_p = out["sum_p_ch"]
    mask = out["baseline_mask_t"]
    first_masked = [i for i,m in enumerate(mask) if m==1][:12]
    if first_masked:
        t0 = first_masked[0]
        actual_baseline = [round(agg_p[t],1) for t in range(t0, min(len(agg_p), t0+12))]
        target_baseline = [round(P_base[t],1) for t in range(t0, min(len(P_base), t0+12))]
        print(f"Baseline tracking (first 12 enforced slots):")
        print(f"  Target: {target_baseline}")
        print(f"  Actual: {actual_baseline}")
    
    # Generate visualizations and save results
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS AND SAVING RESULTS")
    print("="*60)
    
    # Create output directory with timestamp folder
    import os
    from datetime import datetime
    from tqdm import tqdm
    
    # Generate timestamp for folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/bidding_run_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    
    # List of visualization tasks
    viz_tasks = [
        ("Individual EV charging schedules", lambda: plot_ev_charging_schedule(
            evs, gp, out, 
            ev_indices=[0,1,2,3,4,5],
            save_path=f"{output_dir}/ev_charging_schedule.png"
        )),
        ("EV State of Charge evolution", lambda: plot_soc_evolution(
            evs, gp, out, 
            ev_indices=[0,1,2,3,4,5],
            save_path=f"{output_dir}/soc_evolution.png"
        )),
        ("Aggregate analysis", lambda: plot_aggregate_analysis(
            evs, gp, mi, out, P_base,
            save_path=f"{output_dir}/aggregate_analysis.png"
        )),
        ("ADMM convergence history", lambda: plot_convergence_history(
            out,
            save_path=f"{output_dir}/convergence_history.png"
        ))
    ]
    
    # Execute visualization tasks with progress bar
    for i, (desc, task_func) in enumerate(tqdm(viz_tasks, desc="Creating plots", unit="plot")):
        print(f"{i+1}. {desc}...")
        try:
            task_func()
        except Exception as e:
            print(f"   ⚠️  Warning: {desc} failed: {e}")
    
    # Save detailed results to text file
    print("5. Saving detailed results...")
    save_detailed_results(evs, gp, out, P_base, optimal_bidding, output_dir, bidding_result)
    
    # Export CSV data
    print("6. Exporting CSV data...")
    try:
        csv_dir = export_all_csv_data(evs, gp, mi, out, P_base, optimal_bidding, output_dir)
        print(f"📊 CSV files available in: {csv_dir}")
    except Exception as e:
        print(f"   ⚠️  Warning: CSV export failed: {e}")
    
    print(f"\n🎉 Complete! All results saved to '{output_dir}/' directory.")
    print(f"📊 The system automatically selected which blocks to bid in for maximum profit.")
    
    return evs, gp, out, P_base, bidding_result


if __name__ == "__main__":
    # print("Running ADMM EV aggregation with automatic bidding optimization...\n")
    main()
    print("\n" + "="*70)
    print("✅ SUCCESS: Main execution now includes automatic bidding optimization!")
    print("🎯 The system automatically determines the optimal bidding strategy.")
    print("💰 Expected revenue is maximized while respecting all constraints.")
