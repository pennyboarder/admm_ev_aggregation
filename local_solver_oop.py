# -*- coding: utf-8 -*-
"""
Local solver using EVAgent class methods
"""

from typing import List, Dict
import pulp
from models import EVAgent, GlobalParams, MarketInputs


def solve_single_ev_parallel_oop(args):
    """Wrapper function for parallel EV optimization using OOP solver"""
    (i, ev_data, mi, gp, c_r, c_e, c_p, T, B, dt_hr, rho_r, rho_e, rho_p,
     baseline_mask_t, ev_baseline, enforce_baseline_for_reserve, winning_blocks, blocks,
     time_limit_sec_local, msg_local) = args
    
    res = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r, c_e, c_p, T, B, dt_hr, rho_r, rho_e, rho_p,
        baseline_mask_t, P_base=ev_baseline, enforce_baseline_for_reserve=enforce_baseline_for_reserve,
        winning_blocks=winning_blocks, blocks=blocks,
        time_limit_sec=time_limit_sec_local, msg=msg_local
    )
    
    return i, res


def solve_ev_with_agent_methods(
    ev_data, mi, gp, c_r, c_e, c_p, T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
    P_base=None, enforce_baseline_for_reserve=False,
    winning_blocks=None, blocks=None,  # 新しいパラメータ: 落札予定ブロック情報
    use_l2=False, rho_r_l2=0.0, rho_e_l2=0.0, rho_p_l2=0.0, num_segments=5,
    time_limit_sec=5, msg=0
):
    """
    Solve single EV optimization using EVAgent methods
    
    Args:
        ev_data: EV data dictionary
        mi: Market incentive signals
        gp: Grid price signals
        c_r, c_e, c_p: ADMM consensus variables
        rho_r, rho_e, rho_p: ADMM penalty parameters
        baseline_mask_t: Baseline enforcement mask
        P_base: Baseline power per time slot [kW] (optional)
        enforce_baseline_for_reserve: If True, reserve provision requires baseline adherence
        winning_blocks: List of block indices where coordinator expects to win bids
        blocks: Time slot groupings per block for targeted baseline control
        use_l2: Whether to use L2 regularization
        rho_r_l2, rho_e_l2, rho_p_l2: L2 regularization parameters
        num_segments: Number of segments for L2 approximation
        time_limit_sec: Solver time limit
        msg: Solver verbosity
        
    Returns:
        Dictionary with optimization results
    """
    # Create EVAgent instance from data
    ev = EVAgent(
        name=str(ev_data['id']),
        e_kwh=ev_data['battery_capacity'],
        soc0=ev_data['initial_soc'] / 100.0,  # Convert % to fraction
        soc_min=0.1,  # 10% minimum SOC
        soc_max=ev_data['target_soc'] / 100.0,  # Convert % to fraction
        p_charge_max=ev_data['charge_power_max'],
        eta_c=ev_data['charge_efficiency'],
        available=ev_data.get('available', [1] * T),  # Use original EV availability
        discharge_schedule=None,
        eta_d=ev_data['discharge_efficiency'],
        can_discharge_to_grid=ev_data.get('can_discharge', False)
    )

    
    # Use actual blocks structure if provided, otherwise create default blocks
    if blocks is None:
        # Create default blocks (each block is 6 time slots = 30 minutes)
        slots_per_block = 6
        blocks = []
        for b in range(B):
            start_slot = b * slots_per_block
            end_slot = min((b + 1) * slots_per_block, T)
            blocks.append(list(range(start_slot, end_slot)))
    
    # Create post windows for reserve recovery
    post_windows = []
    for block_slots in blocks:
        if len(block_slots) > 0:
            # Post window starts after the block ends
            post_start = max(block_slots) + 1
            post_end = min(post_start + 6, T)  # 6 slots = 30 min recovery
            if post_start < T:
                post_windows.append(list(range(post_start, post_end)))
            else:
                post_windows.append([])
        else:
            post_windows.append([])

    # Create optimization model
    model = pulp.LpProblem(f"EV_{ev.name}_ADMM_OOP", pulp.LpMaximize)

    # Create variables using EV agent method
    ev.create_optimization_variables(T, B)
    
    # Add L2 approximation variables if requested
    if use_l2:
        ev.add_l2_approximation_variables(T, B, num_segments)

    # Add constraints using EV agent methods
    ev.add_power_constraints(model, T)
    ev.add_soc_constraints(model, T, dt_hr)
    
    # Add reserve constraints and ADMM constraints (NOT skipped!)
    ev.add_reserve_constraints(model, blocks, post_windows, dt_hr)
    ev.add_l1_constraints(model, c_r, c_e, c_p, T, B)
    
    # Add baseline constraints based on coordinator strategy
    if P_base is not None:
        if winning_blocks is not None and blocks is not None and len(winning_blocks) > 0:
            # Use targeted baseline constraints for specific winning blocks
            # print(f"🎯 Using targeted baseline control for blocks: {winning_blocks}")
            ev.add_targeted_baseline_constraints(model, P_base, winning_blocks, blocks, T)
        elif enforce_baseline_for_reserve:
            # Use traditional baseline constraints
            # print("📊 Using traditional baseline enforcement")
            ev.add_baseline_constraints(model, P_base, baseline_mask_t, T, B)
    
    # Add L2 approximation constraints if requested
    if use_l2:
        ev.add_l2_approximation_constraints(model, c_r, c_e, c_p, T, B, num_segments)

    # Create objective using EV agent method (includes ADMM terms)
    # Create baseline mask - all time slots are baseline-enabled for targeted control
    baseline_mask_t = [1.0] * T  
    objective = ev.create_objective(mi, gp, T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t)
    model += objective

    # Solve the model
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    model.solve(solver)

    # Extract and return results using EV agent method
    results = ev.extract_results()
    results["model_status"] = pulp.LpStatus[model.status]
    
    return results


# Keep the original function as fallback
def solve_local_ev_cut_with_baseline_L1(
    ev: EVAgent,
    gp: GlobalParams,
    mi: MarketInputs,
    blocks: List[List[int]],
    post_windows: List[List[int]],
    c_r: List[float],
    rho_r: float,
    c_e: List[float],
    rho_e: float,
    c_p: List[float],
    rho_p: float,
    baseline_mask_t: List[int],
    time_limit_sec: int = 6,
    msg: bool = False
) -> Dict:
    """
    Original solver function - now uses the new EVAgent methods
    """
    return solve_ev_with_agent_methods(
        ev, gp, mi, blocks, post_windows,
        c_r, rho_r, c_e, rho_e, c_p, rho_p,
        baseline_mask_t, time_limit_sec, msg
    )
