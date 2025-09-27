# -*- coding: utf-8 -*-
"""
Test script for new EVAgent class with OOP constraints and optimization
"""

from models import EVAgent, GlobalParams, MarketInputs
from local_solver_oop import solve_ev_with_agent_methods
import pulp


def test_ev_agent_oop():
    """Test the new EVAgent class with OOP methods"""
    print("Testing EVAgent Class with Object-Oriented Design")
    print("=" * 60)
    
    # Create a simple test EV
    ev = EVAgent(
        name="TestEV01",
        e_kwh=50.0,
        soc0=0.3,
        soc_min=0.2,
        soc_max=0.9,
        p_charge_max=7.0,
        eta_c=0.95,
        available=[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],  # 12 time slots
        discharge_schedule=[0, 0, 0, 0, 5.0, 3.0, 0, 0, 0, 0, 0, 0],  # Some discharge
        eta_d=0.93,
        can_discharge_to_grid=True
    )
    
    print(f"Created EV: {ev}")
    print(f"Initial SOC: {ev.soc0:.1%}")
    print(f"V2G capable: {ev.can_discharge_to_grid}")
    print(f"Available slots: {sum(ev.available)}/12")
    print()
    
    # Test variable creation
    T = len(ev.available)
    B = 3  # 3 blocks for testing
    
    print("Testing variable creation...")
    variables = ev.create_optimization_variables(T, B)
    print(f"Created variables: {list(variables.keys())}")
    print(f"Power variables: p_ch ({len(variables['p_ch'])} slots)")
    if 'p_dch' in variables:
        print(f"Discharge variables: p_dch ({len(variables['p_dch'])} slots)")
    print(f"SOC variables: soc ({len(variables['soc'])} slots)")
    print(f"Reserve variables: r_cut ({len(variables['r_cut'])} blocks)")
    print()
    
    # Create a simple optimization model to test constraints
    print("Testing constraint creation...")
    model = pulp.LpProblem("TestEV_OOP", pulp.LpMaximize)
    
    # Test power constraints
    power_constraints = ev.add_power_constraints(model, T)
    print(f"Added {len(power_constraints)} power constraints")
    
    # Test SOC constraints
    dt_hr = 5.0 / 60.0  # 5 minutes
    soc_constraints = ev.add_soc_constraints(model, T, dt_hr)
    print(f"Added {len(soc_constraints)} SOC constraints")
    
    # Test simple blocks and post windows
    blocks = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    post_windows = [[], [], []]  # No recovery windows for simplicity
    
    reserve_constraints = ev.add_reserve_constraints(model, blocks, post_windows, dt_hr)
    print(f"Added {len(reserve_constraints)} reserve constraints")
    
    # Test L1 constraints
    c_r = [10.0, 15.0, 5.0]  # Consensus variables
    c_e = [2.0, 3.0, 1.0]
    c_p = [5.0] * T
    
    l1_constraints = ev.add_l1_constraints(model, c_r, c_e, c_p, T, B)
    print(f"Added {len(l1_constraints)} L1 constraints")
    print()
    
    # Test objective creation
    print("Testing objective creation...")
    mi = MarketInputs(
        energy_buy_price_per_kwh=[25.0] * T,
        cap_price_cut_kw_per_block=[100.0] * B
    )
    gp = GlobalParams(dt_min=5, degr_cost_per_kwh=1.0)
    baseline_mask_t = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    
    objective = ev.create_objective(
        mi, gp, T, B, dt_hr,
        rho_r=3.0, rho_e=2.0, rho_p=1.0,
        baseline_mask_t=baseline_mask_t
    )
    model += objective
    print("Created objective function")
    print()
    
    # Solve the model
    print("Solving optimization model...")
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)
    
    print(f"Solver status: {pulp.LpStatus[status]}")
    
    if status == pulp.LpStatusOptimal:
        # Extract results
        results = ev.extract_results()
        print("Optimization successful!")
        print(f"Status: {results['status']}")
        print(f"Reserve capacity: {results['r_cut']}")
        print(f"Recovery energy: {results['e_rec']}")
        p_ch_str = [f"{p:.2f}" for p in results['p_ch'][:6]]
        p_dch_str = [f"{p:.2f}" for p in results['p_dch'][:6]]
        soc_str = [f"{s:.2f}" for s in results['soc'][:6]]
        print(f"Charging power: {p_ch_str}...")
        print(f"Discharging power: {p_dch_str}...")
        print(f"SOC evolution: {soc_str}...")
    else:
        print("Optimization failed!")
    
    print()
    print("EVAgent OOP Test Completed!")
    return ev, results if status == pulp.LpStatusOptimal else None


def test_integrated_solver():
    """Test the integrated OOP solver"""
    print("\n" + "=" * 60)
    print("Testing Integrated OOP Solver")
    print("=" * 60)
    
    # Create test data
    ev = EVAgent(
        name="IntegratedTestEV",
        e_kwh=40.0,
        soc0=0.5,
        soc_min=0.2,
        soc_max=0.9,
        p_charge_max=6.0,
        eta_c=0.95,
        available=[1] * 8,  # 8 time slots all available
        can_discharge_to_grid=True
    )
    
    gp = GlobalParams(dt_min=15, degr_cost_per_kwh=0.5)  # 15-minute intervals
    mi = MarketInputs(
        energy_buy_price_per_kwh=[30.0, 35.0, 25.0, 20.0, 25.0, 30.0, 35.0, 40.0],
        cap_price_cut_kw_per_block=[150.0, 120.0]
    )
    
    blocks = [[0, 1, 2, 3], [4, 5, 6, 7]]
    post_windows = [[], []]
    
    # ADMM parameters
    c_r = [8.0, 6.0]
    c_e = [2.0, 1.5]
    c_p = [4.0] * 8
    baseline_mask_t = [1, 1, 0, 0, 1, 1, 0, 0]
    
    print(f"Testing EV: {ev}")
    print("Running integrated OOP solver...")
    
    results = solve_ev_with_agent_methods(
        ev, gp, mi, blocks, post_windows,
        c_r, 3.0, c_e, 2.0, c_p, 1.0,
        baseline_mask_t,
        time_limit_sec=5, msg=False
    )
    
    print(f"Solver status: {results.get('model_status', 'Unknown')}")
    if results.get('model_status') == 'Optimal':
        r_cut_str = [f"{r:.1f}" for r in results['r_cut']]
        e_rec_str = [f"{e:.1f}" for e in results['e_rec']]
        p_ch_str = [f"{p:.1f}" for p in results['p_ch']]
        p_dch_str = [f"{p:.1f}" for p in results['p_dch']]
        print(f"Reserve capacity: {r_cut_str}")
        print(f"Recovery energy: {e_rec_str}")
        print(f"Charging schedule: {p_ch_str}")
        print(f"Discharging schedule: {p_dch_str}")
        
        # Check mutual exclusion
        violations = sum(1 for p_c, p_d in zip(results['p_ch'], results['p_dch']) 
                        if p_c > 0.001 and p_d > 0.001)
        print(f"Mutual exclusion violations: {violations}")
        
        if violations == 0:
            print("✓ Mutual exclusion constraint working correctly!")
        else:
            print("✗ Mutual exclusion constraint violated!")
    
    return results


if __name__ == "__main__":
    test_ev_agent_oop()
    test_integrated_solver()
