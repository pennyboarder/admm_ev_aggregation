# -*- coding: utf-8 -*-
"""
Visualization functions for EV charging optimization results
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from typing import List, Dict
from models import EVAgent, GlobalParams, MarketInputs


def plot_ev_charging_schedule(
    evs: List[EVAgent],
    gp: GlobalParams,
    results: Dict,
    ev_indices: List[int] = None,
    save_path: str = None,
    show_blocks: bool = True
):
    """
    Plot individual EV charging schedules
    
    Args:
        evs: List of EV agents
        gp: Global parameters
        results: ADMM optimization results
        ev_indices: Which EVs to plot (default: first 6)
        save_path: Path to save the plot
        show_blocks: Whether to show block boundaries
    """
    if ev_indices is None:
        ev_indices = list(range(min(6, len(evs))))
    
    n_evs = len(ev_indices)
    T = len(results['sum_p_ch'])
    dt_hr = gp.dt_min / 60.0
    
    # Time axis in hours
    time_hours = np.arange(T) * dt_hr
    
    fig, axes = plt.subplots(n_evs, 1, figsize=(12, 2*n_evs), sharex=True)
    if n_evs == 1:
        axes = [axes]
    
    # Get individual charging powers (need to re-solve to get individual results)
    individual_results = get_individual_charging_schedules(evs, gp, results)
    
    for idx, ev_idx in enumerate(ev_indices):
        ev = evs[ev_idx]
        p_ch = individual_results[ev_idx]['p_ch']
        
        # Plot charging power
        axes[idx].step(time_hours, p_ch, where='post', linewidth=2, label='Charging Power')
        axes[idx].fill_between(time_hours, 0, p_ch, step='post', alpha=0.3)
        
        # Plot availability
        availability = np.array(ev.available) * ev.p_charge_max * 0.9  # Show as 90% of max for visibility
        axes[idx].step(time_hours, availability, where='post', color='gray', alpha=0.5, linestyle='--', label='Availability')
        
        # Show block boundaries
        if show_blocks:
            blocks = results['blocks']
            for b, block_slots in enumerate(blocks):
                if block_slots:
                    block_start = block_slots[0] * dt_hr
                    axes[idx].axvline(x=block_start, color='red', alpha=0.3, linestyle=':')
        
        axes[idx].set_ylabel(f'Power [kW]\n{ev.name}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)
        axes[idx].set_ylim(0, max(ev.p_charge_max * 1.1, max(p_ch) * 1.1))
    
    axes[-1].set_xlabel('Time [hours]')
    plt.suptitle('Individual EV Charging Schedules', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved EV charging schedule plot to: {save_path}")
    plt.show(block=False)  # Don't block execution
    plt.pause(0.1)  # Brief pause to render


def plot_soc_evolution(
    evs: List[EVAgent],
    gp: GlobalParams,
    results: Dict,
    ev_indices: List[int] = None,
    save_path: str = None
):
    """
    Plot SOC (State of Charge) evolution for selected EVs
    """
    if ev_indices is None:
        ev_indices = list(range(min(6, len(evs))))
    
    n_evs = len(ev_indices)
    T = len(results['sum_p_ch'])
    dt_hr = gp.dt_min / 60.0
    time_hours = np.arange(T) * dt_hr
    
    individual_results = get_individual_charging_schedules(evs, gp, results)
    
    fig, axes = plt.subplots(n_evs, 1, figsize=(12, 2*n_evs), sharex=True)
    if n_evs == 1:
        axes = [axes]
    
    for idx, ev_idx in enumerate(ev_indices):
        ev = evs[ev_idx]
        p_ch = individual_results[ev_idx]['p_ch']
        
        # Calculate SOC evolution with discharge
        soc = np.zeros(T)
        soc[0] = ev.soc0
        
        # Handle first slot
        charge_energy_0 = p_ch[0] * ev.eta_c * dt_hr / ev.e_kwh if len(p_ch) > 0 else 0
        external_use_0 = (ev.discharge_schedule[0] / ev.e_kwh) if ev.discharge_schedule and len(ev.discharge_schedule) > 0 else 0
        soc[0] = ev.soc0 + charge_energy_0 - external_use_0
        
        for t in range(1, T):
            charge_energy = p_ch[t-1] * ev.eta_c * dt_hr / ev.e_kwh if t-1 < len(p_ch) else 0
            external_use = (ev.discharge_schedule[t-1] / ev.e_kwh) if ev.discharge_schedule and t-1 < len(ev.discharge_schedule) else 0
            soc[t] = soc[t-1] + charge_energy - external_use
            # Clamp to valid range
            soc[t] = max(ev.soc_min, min(ev.soc_max, soc[t]))
        
        axes[idx].plot(time_hours, soc * 100, linewidth=2, label='SOC')
        axes[idx].axhline(y=ev.soc_min * 100, color='red', linestyle='--', alpha=0.7, label='Min SOC')
        axes[idx].axhline(y=ev.soc_max * 100, color='orange', linestyle='--', alpha=0.7, label='Max SOC')
        
        axes[idx].set_ylabel(f'SOC [%]\n{ev.name}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)
        axes[idx].set_ylim(0, 100)
    
    axes[-1].set_xlabel('Time [hours]')
    plt.suptitle('EV State of Charge Evolution', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved SOC evolution plot to: {save_path}")
    plt.show(block=False)  # Don't block execution
    plt.pause(0.1)  # Brief pause to render


def plot_aggregate_analysis(
    evs: List[EVAgent],
    gp: GlobalParams,
    mi: MarketInputs,
    results: Dict,
    P_base: List[float],
    save_path: str = None
):
    """
    Plot aggregate charging power, reserve capacity, and market prices
    """
    T = len(results['sum_p_ch'])
    B = len(results['sum_r_cut'])
    dt_hr = gp.dt_min / 60.0
    
    time_hours = np.arange(T) * dt_hr
    block_hours = np.arange(B) * gp.sustain_min / 60.0
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 1. Aggregate charging power vs baseline
    axes[0].step(time_hours, results['sum_p_ch'], where='post', linewidth=2, label='Actual Aggregate Charging', color='blue')
    axes[0].step(time_hours, P_base, where='post', linewidth=2, linestyle='--', label='Baseline', color='red')
    
    # Highlight enforced blocks
    mask = results['baseline_mask_t']
    enforced_power = [results['sum_p_ch'][t] if mask[t] else 0 for t in range(T)]
    axes[0].fill_between(time_hours, 0, enforced_power, step='post', alpha=0.3, color='green', label='Enforced Blocks')
    
    axes[0].set_ylabel('Power [kW]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Aggregate Charging Power')
    
    # 2. Reserve capacity (r_cut)
    axes[1].step(block_hours, results['sum_r_cut'], where='post', linewidth=2, color='orange', marker='o')
    axes[1].fill_between(block_hours, 0, results['sum_r_cut'], step='post', alpha=0.3, color='orange')
    axes[1].set_ylabel('Reserve [kW]')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Up-Reserve Capacity (Charging Stop)')
    
    # 3. Recovery energy
    axes[2].step(block_hours, results['sum_e_rec'], where='post', linewidth=2, color='purple', marker='s')
    axes[2].fill_between(block_hours, 0, results['sum_e_rec'], step='post', alpha=0.3, color='purple')
    axes[2].set_ylabel('Energy [kWh]')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Recovery Energy After Reserve Activation')
    
    # 4. Market prices
    axes[3].step(time_hours, mi.energy_buy_price_per_kwh, where='post', linewidth=2, color='green', label='Energy Price')
    
    # Reserve prices (expand to time resolution)
    reserve_prices_expanded = []
    blocks = results['blocks']
    for t in range(T):
        for b, block_slots in enumerate(blocks):
            if t in block_slots:
                reserve_prices_expanded.append(mi.cap_price_cut_kw_per_block[b])
                break
    
    ax3_twin = axes[3].twinx()
    ax3_twin.step(time_hours, reserve_prices_expanded, where='post', linewidth=2, color='red', label='Reserve Price')
    ax3_twin.set_ylabel('Reserve Price [¥/kW]', color='red')
    
    axes[3].set_ylabel('Energy Price [¥/kWh]', color='green')
    axes[3].set_xlabel('Time [hours]')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Market Prices')
    
    # Add legends
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregate analysis plot to: {save_path}")
    plt.show(block=False)  # Don't block execution
    plt.pause(0.1)  # Brief pause to render


def plot_convergence_history(results: Dict, save_path: str = None):
    """
    Plot ADMM convergence history
    """
    history = results['history']
    if not history:
        print("No convergence history available")
        return
    
    iterations = [h['iter'] for h in history]
    r_cut_res = [h['r_cut_res'] for h in history]
    e_rec_res = [h['e_rec_res'] for h in history]
    base_res = [h['base_res'] for h in history]
    total_res = [h['r_norm'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Individual residuals
    ax1.semilogy(iterations, r_cut_res, 'o-', label='Reserve Residual', alpha=0.8)
    ax1.semilogy(iterations, e_rec_res, 's-', label='Recovery Residual', alpha=0.8)
    ax1.semilogy(iterations, base_res, '^-', label='Baseline Residual', alpha=0.8)
    ax1.set_ylabel('Residual (log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Individual Constraint Residuals')
    
    # Total residual
    ax2.semilogy(iterations, total_res, 'o-', color='red', linewidth=2)
    ax2.set_xlabel('ADMM Iteration')
    ax2.set_ylabel('Total Residual (log scale)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Total Residual Convergence')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence history plot to: {save_path}")
    plt.show(block=False)  # Don't block execution
    plt.pause(0.1)  # Brief pause to render


def get_individual_charging_schedules(evs: List[EVAgent], gp: GlobalParams, results: Dict) -> List[Dict]:
    """
    Extract individual EV charging schedules from ADMM results
    """
    if 'individual_p_ch' in results:
        # Use actual individual results from ADMM
        return [{'p_ch': results['individual_p_ch'][i]} for i in range(len(evs))]
    else:
        # Fallback: proportional allocation (for compatibility)
        T = len(results['sum_p_ch'])
        N = len(evs)
        individual_results = []
        
        for i, ev in enumerate(evs):
            p_ch = []
            for t in range(T):
                total_avail_at_t = sum(evs[j].p_charge_max * evs[j].available[t] for j in range(N))
                if total_avail_at_t > 0:
                    share = (ev.p_charge_max * ev.available[t]) / total_avail_at_t
                    p_ch.append(results['sum_p_ch'][t] * share)
                else:
                    p_ch.append(0.0)
            
            individual_results.append({'p_ch': p_ch})
        
        return individual_results


def plot_ev_discharge_patterns(
    evs: List[EVAgent],
    gp: GlobalParams,
    results: Dict,
    ev_indices: List[int] = None,
    save_path: str = None
):
    """
    Plot EV discharge patterns (external use) and net power flow
    """
    if ev_indices is None:
        ev_indices = list(range(min(6, len(evs))))
    
    n_evs = len(ev_indices)
    
    # Handle case when no EVs to plot
    if n_evs == 0:
        print("No EVs with discharge capability found for plotting.")
        # Create empty plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.text(0.5, 0.5, 'No EVs with discharge capability', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 24)
        ax.set_xlabel('Time [hours]')
        ax.set_title('EV Discharge Patterns')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved empty discharge pattern plot to: {save_path}")
        plt.close()
        return
    
    T = len(results['sum_p_ch'])
    dt_hr = gp.dt_min / 60.0
    time_hours = np.arange(T) * dt_hr
    
    individual_results = get_individual_charging_schedules(evs, gp, results)
    
    fig, axes = plt.subplots(n_evs, 1, figsize=(12, 2*n_evs), sharex=True)
    if n_evs == 1:
        axes = [axes]
    
    for idx, ev_idx in enumerate(ev_indices):
        ev = evs[ev_idx]
        p_ch = individual_results[ev_idx]['p_ch']
        
        # Plot charging (positive)
        axes[idx].step(time_hours, p_ch, where='post', linewidth=2, 
                      color='green', label='Charging')
        axes[idx].fill_between(time_hours, 0, p_ch, step='post', 
                              alpha=0.3, color='green')
        
        # Plot external discharge (negative)
        if ev.discharge_schedule:
            # Convert kWh to equivalent power
            discharge_power = [d / dt_hr for d in ev.discharge_schedule[:T]]
            axes[idx].step(time_hours, [-p for p in discharge_power], where='post', 
                          linewidth=2, color='red', label='External Use')
            axes[idx].fill_between(time_hours, 0, [-p for p in discharge_power], 
                                  step='post', alpha=0.3, color='red')
        
        # Plot net power (charging - external use)
        net_power = p_ch[:]
        if ev.discharge_schedule:
            for t in range(min(len(net_power), len(ev.discharge_schedule))):
                net_power[t] -= ev.discharge_schedule[t] / dt_hr
        
        axes[idx].step(time_hours, net_power, where='post', linewidth=2, 
                      color='blue', linestyle='--', alpha=0.8, label='Net Power')
        
        # Reference lines
        axes[idx].axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        
        axes[idx].set_ylabel(f'Power [kW]\n{ev.name}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)
        
        # Set y-limits to show both positive and negative values
        max_val = max(max(p_ch), ev.p_charge_max * 1.1)
        if ev.discharge_schedule:
            min_val = -max(ev.discharge_schedule[:T]) / dt_hr * 1.1
        else:
            min_val = 0
        axes[idx].set_ylim(min_val, max_val)
    
    axes[-1].set_xlabel('Time [hours]')
    plt.suptitle('EV Charging and Discharge Patterns', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved EV discharge patterns plot to: {save_path}")
    plt.show(block=False)
    plt.pause(0.1)


def save_all_plots(
    evs: List[EVAgent],
    gp: GlobalParams,
    mi: MarketInputs,
    results: Dict,
    P_base: List[float],
    output_dir: str = "output",
    prefix: str = None,
    ev_indices: List[int] = None
):
    """
    Save all visualization plots to specified directory
    
    Args:
        evs: List of EV agents
        gp: Global parameters
        mi: Market inputs
        results: ADMM optimization results
        P_base: Baseline power profile
        output_dir: Directory to save plots
        prefix: Prefix for filenames
        ev_indices: Which EVs to plot (default: first 4)
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate timestamp and prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix is None:
        # If output_dir doesn't contain timestamp, create timestamped subdirectory
        if "run_" not in output_dir:
            output_dir = f"{output_dir}/run_{timestamp}"
        prefix = f"ev_optimization"
    else:
        if "run_" not in output_dir:
            output_dir = f"{output_dir}/run_{timestamp}"
        prefix = f"{prefix}"
    
    if ev_indices is None:
        ev_indices = list(range(min(4, len(evs))))
    
    print(f"Saving all plots to: {output_dir}")
    print(f"Filename prefix: {prefix}")
    
    # Ensure output directory exists before saving files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Save all plots
    plot_ev_charging_schedule(
        evs, gp, results, 
        ev_indices=ev_indices,
        save_path=f"{output_dir}/{prefix}_charging_schedule.png"
    )
    
    plot_soc_evolution(
        evs, gp, results, 
        ev_indices=ev_indices,
        save_path=f"{output_dir}/{prefix}_soc_evolution.png"
    )
    
    plot_aggregate_analysis(
        evs, gp, mi, results, P_base,
        save_path=f"{output_dir}/{prefix}_aggregate_analysis.png"
    )
    
    plot_convergence_history(
        results,
        save_path=f"{output_dir}/{prefix}_convergence_history.png"
    )
    
    print(f"All plots saved successfully!")
    return {
        'charging_schedule': f"{output_dir}/{prefix}_charging_schedule.png",
        'soc_evolution': f"{output_dir}/{prefix}_soc_evolution.png", 
        'aggregate_analysis': f"{output_dir}/{prefix}_aggregate_analysis.png",
        'convergence_history': f"{output_dir}/{prefix}_convergence_history.png"
    }
