"""
Enhanced solver with bidding decision optimization
"""
from typing import List, Dict
import pulp
from models import EVAgent


def solve_ev_with_bidding_optimization(
    ev_data, mi, gp, blocks, T, B, dt_hr, 
    R_max_per_block: List[float],
    cap_prices: List[float],
    P_base: List[float] = None,
    time_limit_sec: int = 10,
    msg: int = 0
) -> Dict:
    """
    Solve EV optimization with endogenous bidding decisions
    
    Args:
        ev_data: EV data dictionary
        mi: Market inputs
        gp: Global parameters  
        blocks: Time slot groupings per block
        T: Number of time slots
        B: Number of blocks
        dt_hr: Time step in hours
        R_max_per_block: Maximum reserve capacity per block [kW]
        cap_prices: Capacity prices per block [¥/kW]
        P_base: Baseline power per slot (if None, calculated optimally)
        time_limit_sec: Solver time limit
        msg: Solver verbosity
        
    Returns:
        Dictionary with optimization results including bidding decisions
    """
    # Create EVAgent instance
    ev = EVAgent(
        name=str(ev_data['id']),
        e_kwh=ev_data['battery_capacity'],
        soc0=ev_data['initial_soc'] / 100.0,
        soc_min=0.1,
        soc_max=ev_data['target_soc'] / 100.0,
        p_charge_max=ev_data['charge_power_max'],
        eta_c=ev_data['charge_efficiency'],
        available=[1] * T,
        eta_d=ev_data['discharge_efficiency'],
        can_discharge_to_grid=ev_data.get('can_discharge', False)
    )
    
    # Create optimization model
    model = pulp.LpProblem(f"EV_Bidding_Optimization_{ev.name}", pulp.LpMaximize)
    
    # Create variables
    ev.create_optimization_variables(T, B)
    
    # Add bidding decision variables
    ev.add_bidding_decision_variables(B)
    
    # Add basic constraints
    ev.add_power_constraints(model, T)
    ev.add_soc_constraints(model, T, dt_hr)
    
    # Add conditional constraints based on bidding decisions
    if P_base is not None:
        ev.add_conditional_baseline_constraints(model, P_base, blocks, T, B)
    
    ev.add_conditional_reserve_constraints(model, R_max_per_block, B)
    
    # Enhanced objective function with bidding considerations
    variables = ev.variables
    
    # Energy costs
    energy_cost = pulp.lpSum(
        mi.energy_buy_price_per_kwh[t] * variables['p_ch'][t] * dt_hr
        for t in range(T)
    )
    
    # Reserve revenue (linearized with auxiliary variables)
    # Need auxiliary variables: revenue_b = cap_prices[b] * r_cut[b] * bid_decision[b]
    revenue_vars = {}
    reserve_revenue = 0.0
    
    for b in range(B):
        # Create auxiliary variable for revenue from block b
        revenue_vars[b] = pulp.LpVariable(f"revenue_{b}", lowBound=0.0)
        
        # Linearization constraints:
        # revenue_b <= M * bid_decision[b]
        # revenue_b <= cap_prices[b] * r_cut[b]
        # revenue_b >= cap_prices[b] * r_cut[b] - M * (1 - bid_decision[b])
        M_revenue = cap_prices[b] * R_max_per_block[b]  # Big-M
        
        model += revenue_vars[b] <= M_revenue * variables['bid_decision'][b]
        model += revenue_vars[b] <= cap_prices[b] * variables['r_cut'][b]
        model += revenue_vars[b] >= cap_prices[b] * variables['r_cut'][b] - M_revenue * (1 - variables['bid_decision'][b])
        
        reserve_revenue += revenue_vars[b]
    
    # Bidding costs (baseline compliance costs)
    baseline_cost = 0.0
    if P_base is not None:
        # Penalty for deviation from optimal charging (simplified)
        baseline_cost = pulp.lpSum(
            variables['bid_decision'][b] * 5.0  # Fixed cost per bidding block
            for b in range(B)
        )
    
    # SOC target achievement reward
    soc_target_reward = 1000.0 * variables['soc'][T-1]  # Reward for high final SOC
    
    # Objective: maximize profit
    objective = reserve_revenue - energy_cost - baseline_cost + soc_target_reward
    model += objective
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    model.solve(solver)
    
    # Extract results
    if model.status == 1:  # Optimal
        results = ev.extract_results()
        results["model_status"] = "Optimal"
        results["objective_value"] = pulp.value(objective)
        
        # Calculate profitability metrics
        if 'bid_decision' in results:
            total_bidding_blocks = sum(results['bid_decision'])
            total_reserve_revenue = sum(
                cap_prices[b] * results['r_cut'][b] * results['bid_decision'][b]
                for b in range(B)
            )
            total_energy_cost = sum(
                mi.energy_buy_price_per_kwh[t] * results['p_ch'][t] * dt_hr
                for t in range(T)
            )
            
            results["bidding_metrics"] = {
                "total_bidding_blocks": total_bidding_blocks,
                "total_reserve_revenue": total_reserve_revenue,
                "total_energy_cost": total_energy_cost,
                "net_profit": total_reserve_revenue - total_energy_cost
            }
        
        return results
    else:
        return {
            "model_status": pulp.LpStatus[model.status],
            "status": "Failed"
        }


def solve_ev_with_fixed_bidding(
    ev_data, mi, gp, blocks, T, B, dt_hr,
    R_max_per_block: List[float],
    cap_prices: List[float], 
    enforce_blocks: List[int],
    P_base: List[float],
    time_limit_sec: int = 10,
    msg: int = 0
) -> Dict:
    """
    Solve EV optimization with fixed bidding decisions (for comparison)
    """
    # Create EVAgent instance
    ev = EVAgent(
        name=str(ev_data['id']),
        e_kwh=ev_data['battery_capacity'],
        soc0=ev_data['initial_soc'] / 100.0,
        soc_min=0.1,
        soc_max=ev_data['target_soc'] / 100.0,
        p_charge_max=ev_data['charge_power_max'],
        eta_c=ev_data['charge_efficiency'],
        available=[1] * T,
        eta_d=ev_data['discharge_efficiency'],
        can_discharge_to_grid=ev_data.get('can_discharge', False)
    )
    
    # Create model and variables
    model = pulp.LpProblem(f"EV_Fixed_Bidding_{ev.name}", pulp.LpMaximize)
    ev.create_optimization_variables(T, B)
    
    # Add constraints
    ev.add_power_constraints(model, T)
    ev.add_soc_constraints(model, T, dt_hr)
    
    # Fixed baseline constraints
    variables = ev.variables
    for b, enforce in enumerate(enforce_blocks):
        if enforce and b < len(blocks):
            block_slots = blocks[b]
            for t in block_slots:
                if t < T and t < len(P_base):
                    model += variables['p_ch'][t] == P_base[t]
        
        # Fixed reserve limits
        if enforce:
            model += variables['r_cut'][b] <= R_max_per_block[b]
        else:
            model += variables['r_cut'][b] == 0.0
    
    # Objective
    energy_cost = pulp.lpSum(
        mi.energy_buy_price_per_kwh[t] * variables['p_ch'][t] * dt_hr
        for t in range(T)
    )
    
    reserve_revenue = pulp.lpSum(
        cap_prices[b] * variables['r_cut'][b] * enforce_blocks[b]
        for b in range(B)
    )
    
    soc_target_reward = 1000.0 * variables['soc'][T-1]
    
    objective = reserve_revenue - energy_cost + soc_target_reward
    model += objective
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    model.solve(solver)
    
    # Extract results
    if model.status == 1:
        results = ev.extract_results()
        results["model_status"] = "Optimal"
        results["objective_value"] = pulp.value(objective)
        results["bid_decision"] = enforce_blocks  # Fixed decisions
        
        return results
    else:
        return {
            "model_status": pulp.LpStatus[model.status],
            "status": "Failed"
        }
