# -*- coding: utf-8 -*-
"""
CSV export functions for EV optimization results
"""

import csv
import os
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from models import EVAgent, GlobalParams, MarketInputs


def export_time_series_csv(
    evs: List[EVAgent],
    gp: GlobalParams,
    mi: MarketInputs,
    results: Dict,
    P_base: List[float],
    enforce_blocks: List[int],
    output_dir: str
):
    """
    Export detailed time series data to CSV files
    
    Args:
        evs: List of EV agents
        gp: Global parameters
        mi: Market inputs
        results: ADMM optimization results
        P_base: Baseline power profile
        enforce_blocks: Enforcement mask for blocks
        output_dir: Output directory path
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    T = len(results['sum_p_ch'])
    B = len(results['sum_r_cut'])
    N = len(evs)
    dt_hr = gp.dt_min / 60.0
    
    print("Exporting time series data to CSV files...")
    
    # Define export tasks
    export_tasks = [
        ("Time slot data", lambda: export_timeslot_data(evs, gp, mi, results, P_base, output_dir, T, dt_hr)),
        ("Block data", lambda: export_block_data(results, enforce_blocks, gp, output_dir, B)),
        ("Individual EV data", lambda: export_individual_ev_data(evs, results, gp, output_dir, T, N, dt_hr)),
        ("Market data", lambda: export_market_data(mi, results, gp, output_dir, T, B, dt_hr)),
        ("ADMM convergence", lambda: export_convergence_data(results, output_dir)),
        ("Summary statistics", lambda: export_summary_statistics(evs, results, gp, output_dir))
    ]
    
    # Execute export tasks with progress bar
    for desc, task_func in tqdm(export_tasks, desc="Exporting CSV files", unit="file"):
        task_func()
    
    print(f"CSV export completed! Files saved to: {output_dir}")
    return output_dir


def export_timeslot_data(evs, gp, mi, results, P_base, output_dir, T, dt_hr):
    """Export time slot level aggregate data"""
    
    csv_file = os.path.join(output_dir, "timeslot_data.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'TimeSlot', 'TimeHours', 'HourOfDay',
            'AggregateCharging_kW', 'Baseline_kW', 'BaselineEnforced',
            'EnergyPrice_YenPerKWh', 'AvailableEVs', 'MaxChargingCapacity_kW'
        ])
        
        # Data rows
        for t in range(T):
            time_hours = t * dt_hr
            hour_of_day = int(time_hours) % 24
            
            # Count available EVs and total capacity at this time slot
            available_evs = sum(1 for ev in evs if ev.available[t])
            max_capacity = sum(ev.p_charge_max * ev.available[t] for ev in evs)
            
            baseline_enforced = results['baseline_mask_t'][t]
            
            writer.writerow([
                t,
                f"{time_hours:.2f}",
                hour_of_day,
                f"{results['sum_p_ch'][t]:.3f}",
                f"{P_base[t]:.3f}",
                baseline_enforced,
                f"{mi.energy_buy_price_per_kwh[t]:.2f}",
                available_evs,
                f"{max_capacity:.3f}"
            ])
    
    print(f"  Time slot data: {csv_file}")


def export_block_data(results, enforce_blocks, gp, output_dir, B):
    """Export block level data"""
    
    csv_file = os.path.join(output_dir, "block_data.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Block', 'StartTimeHours', 'EndTimeHours', 'StartHourOfDay',
            'ReserveCapacity_kW', 'RecoveryEnergy_kWh', 'Enforced',
            'ReserveTarget_kW', 'RecoveryTarget_kWh'
        ])
        
        # Data rows
        blocks = results['blocks']
        for b in range(B):
            if b < len(blocks):
                start_slot = blocks[b][0] if blocks[b] else 0
                end_slot = blocks[b][-1] if blocks[b] else 0
                start_time_hours = start_slot * gp.dt_min / 60.0
                end_time_hours = (end_slot + 1) * gp.dt_min / 60.0
                start_hour_of_day = int(start_time_hours) % 24
            else:
                start_time_hours = end_time_hours = start_hour_of_day = 0
            
            # Calculate targets
            reserve_target = results['sum_r_cut'][b] if enforce_blocks[b] else 0
            recovery_target = 0.5 * reserve_target if enforce_blocks[b] else 0
            
            writer.writerow([
                b,
                f"{start_time_hours:.2f}",
                f"{end_time_hours:.2f}",
                start_hour_of_day,
                f"{results['sum_r_cut'][b]:.3f}",
                f"{results['sum_e_rec'][b]:.3f}",
                enforce_blocks[b],
                f"{reserve_target:.3f}",
                f"{recovery_target:.3f}"
            ])
    
    print(f"  Block data: {csv_file}")


def export_individual_ev_data(evs, results, gp, output_dir, T, N, dt_hr):
    """Export individual EV charging schedules and SOC evolution"""
    
    # Check if individual results are available
    if 'individual_p_ch' not in results:
        print("  Individual EV data not available (no individual_p_ch in results)")
        return
    
    csv_file = os.path.join(output_dir, "individual_ev_data.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header - create columns for each EV's charging power, discharging power, and SOC
        header = ['TimeSlot', 'TimeHours', 'HourOfDay']
        for i, ev in enumerate(evs):
            header.extend([f'{ev.name}_Charging_kW', f'{ev.name}_Discharging_kW', f'{ev.name}_Net_kW', f'{ev.name}_SOC_%', f'{ev.name}_Available'])
        writer.writerow(header)
        
        # Calculate SOC evolution for each EV including discharge
        soc_evolution = []
        for i, ev in enumerate(evs):
            soc = [0.0] * T
            soc[0] = ev.soc0
            for t in range(1, T):
                # Energy from charging
                charge_energy = results['individual_p_ch'][i][t-1] * ev.eta_c * dt_hr
                
                # Energy from discharging (if available)
                discharge_energy = 0.0
                if 'individual_p_dch' in results and results['individual_p_dch']:
                    discharge_energy = results['individual_p_dch'][i][t-1] / ev.eta_d * dt_hr
                
                # External use (if defined)
                external_use = 0.0
                if hasattr(ev, 'discharge_schedule') and ev.discharge_schedule and t-1 < len(ev.discharge_schedule):
                    external_use = ev.discharge_schedule[t-1] * dt_hr
                
                # Update SOC
                soc[t] = soc[t-1] + (charge_energy - discharge_energy - external_use) / ev.e_kwh
                
                # Clamp to valid range
                soc[t] = max(ev.soc_min, min(ev.soc_max, soc[t]))
            soc_evolution.append(soc)
        
        # Data rows
        for t in range(T):
            time_hours = t * dt_hr
            hour_of_day = int(time_hours) % 24
            
            row = [t, f"{time_hours:.2f}", hour_of_day]
            
            for i, ev in enumerate(evs):
                charging_power = results['individual_p_ch'][i][t]
                
                # Get discharging power if available
                discharging_power = 0.0
                if 'individual_p_dch' in results and results['individual_p_dch']:
                    discharging_power = results['individual_p_dch'][i][t]
                
                # Calculate net power (positive = charging, negative = discharging)
                net_power = charging_power - discharging_power
                
                soc_percent = soc_evolution[i][t] * 100
                available = ev.available[t]
                
                row.extend([
                    f"{charging_power:.3f}",
                    f"{discharging_power:.3f}",
                    f"{net_power:.3f}",
                    f"{soc_percent:.1f}",
                    available
                ])
            
            writer.writerow(row)
    
    print(f"  Individual EV data: {csv_file}")


def export_market_data(mi, results, gp, output_dir, T, B, dt_hr):
    """Export market pricing and revenue data"""
    
    csv_file = os.path.join(output_dir, "market_data.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'TimeSlot', 'TimeHours', 'HourOfDay', 'Block',
            'EnergyPrice_YenPerKWh', 'ReservePrice_YenPerKW',
            'EnergyConsumption_kWh', 'EnergyCost_Yen',
            'ReserveRevenue_Yen', 'NetValue_Yen'
        ])
        
        # Map time slots to blocks
        blocks = results['blocks']
        slot_to_block = {}
        for b, block_slots in enumerate(blocks):
            for slot in block_slots:
                slot_to_block[slot] = b
        
        # Data rows
        for t in range(T):
            time_hours = t * dt_hr
            hour_of_day = int(time_hours) % 24
            block = slot_to_block.get(t, -1)
            
            energy_consumption = results['sum_p_ch'][t] * dt_hr  # kWh
            energy_cost = energy_consumption * mi.energy_buy_price_per_kwh[t]
            
            if block >= 0 and block < len(mi.cap_price_cut_kw_per_block):
                reserve_price = mi.cap_price_cut_kw_per_block[block]
                reserve_revenue = results['sum_r_cut'][block] * reserve_price / len(blocks[block])  # Distributed over block slots
            else:
                reserve_price = 0
                reserve_revenue = 0
            
            net_value = reserve_revenue - energy_cost
            
            writer.writerow([
                t,
                f"{time_hours:.2f}",
                hour_of_day,
                block,
                f"{mi.energy_buy_price_per_kwh[t]:.2f}",
                f"{reserve_price:.2f}",
                f"{energy_consumption:.4f}",
                f"{energy_cost:.2f}",
                f"{reserve_revenue:.2f}",
                f"{net_value:.2f}"
            ])
    
    print(f"  Market data: {csv_file}")


def export_convergence_data(results, output_dir):
    """Export ADMM convergence history"""
    
    if not results.get('history'):
        print("  ADMM convergence data not available")
        return
    
    csv_file = os.path.join(output_dir, "admm_convergence.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Iteration', 'ReserveResidual', 'RecoveryResidual', 
            'BaselineResidual', 'TotalResidual'
        ])
        
        # Data rows
        for h in results['history']:
            writer.writerow([
                h['iter'],
                f"{h['r_cut_res']:.6f}",
                f"{h['e_rec_res']:.6f}",
                f"{h['base_res']:.6f}",
                f"{h['r_norm']:.6f}"
            ])
    
    print(f"  ADMM convergence: {csv_file}")


def export_summary_statistics(evs, results, gp, output_dir):
    """Export summary statistics"""
    
    csv_file = os.path.join(output_dir, "summary_statistics.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # System metrics
        writer.writerow(['Metric', 'Value', 'Unit'])
        writer.writerow(['Number_of_EVs', len(evs), 'count'])
        writer.writerow(['Time_resolution', gp.dt_min, 'minutes'])
        writer.writerow(['Simulation_duration', len(results['sum_p_ch']) * gp.dt_min / 60, 'hours'])
        writer.writerow(['Number_of_blocks', len(results['sum_r_cut']), 'count'])
        
        # Fleet statistics
        total_capacity = sum(ev.e_kwh for ev in evs)
        avg_capacity = total_capacity / len(evs)
        total_max_power = sum(ev.p_charge_max for ev in evs)
        avg_initial_soc = sum(ev.soc0 for ev in evs) / len(evs)
        
        writer.writerow(['Total_fleet_capacity', f"{total_capacity:.1f}", 'kWh'])
        writer.writerow(['Average_EV_capacity', f"{avg_capacity:.1f}", 'kWh'])
        writer.writerow(['Total_max_charging_power', f"{total_max_power:.1f}", 'kW'])
        writer.writerow(['Average_initial_SOC', f"{avg_initial_soc:.3f}", 'fraction'])
        
        # Optimization results
        total_reserve = sum(results['sum_r_cut'])
        total_recovery = sum(results['sum_e_rec'])
        total_charging = sum(results['sum_p_ch']) * gp.dt_min / 60  # kWh
        
        writer.writerow(['Total_reserve_capacity', f"{total_reserve:.1f}", 'kW'])
        writer.writerow(['Total_recovery_energy', f"{total_recovery:.1f}", 'kWh'])
        writer.writerow(['Total_energy_charged', f"{total_charging:.1f}", 'kWh'])
        
        # Convergence info
        if results.get('history'):
            final_residual = results['history'][-1]['r_norm']
            iterations = len(results['history'])
            writer.writerow(['ADMM_iterations', iterations, 'count'])
            writer.writerow(['Final_residual', f"{final_residual:.6f}", 'dimensionless'])
            writer.writerow(['Converged', 'Yes' if final_residual < 1e-3 else 'No', 'boolean'])
        
        # Time stamp
        writer.writerow(['Export_timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'datetime'])
    
    print(f"  Summary statistics: {csv_file}")


def export_all_csv_data(
    evs: List[EVAgent],
    gp: GlobalParams,
    mi: MarketInputs,
    results: Dict,
    P_base: List[float],
    enforce_blocks: List[int],
    output_dir: str = None,
    prefix: str = None
):
    """
    Convenient function to export all CSV data with automatic directory creation
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/run_{timestamp}"
    
    if prefix:
        csv_subdir = os.path.join(output_dir, f"{prefix}_csv_data")
    else:
        csv_subdir = os.path.join(output_dir, "csv_data")
    
    export_time_series_csv(evs, gp, mi, results, P_base, enforce_blocks, csv_subdir)
    
    return csv_subdir
